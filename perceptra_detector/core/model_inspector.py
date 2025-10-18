"""
Smart model inspection and backend detection.

This module analyzes model files to determine the correct backend
beyond just file extensions.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def inspect_model(model_path: Path) -> Dict[str, Any]:
    """
    Inspect a model file and extract metadata.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model information including suggested backend
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    extension = model_path.suffix.lower()
    
    # Try PyTorch inspection
    if extension in ['.pt', '.pth']:
        return _inspect_pytorch_model(model_path)
    
    # Try ONNX inspection
    elif extension == '.onnx':
        return _inspect_onnx_model(model_path)
    
    else:
        return {
            'backend': None,
            'format': extension,
            'reason': 'Unknown format'
        }

def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """
    import contextlib
    import torch
    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))


    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    logger.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"


def _inspect_pytorch_model(model_path: Path) -> Dict[str, Any]:
    """
    Inspect PyTorch model to determine backend.
    
    Checks for specific model signatures and metadata.
    """
    try:
        import torch
        
        # Load with weights_only=True for security (PyTorch 1.13+)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions
            checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'format': 'pytorch',
            'backend': None,
            'metadata': {}
        }
        
        # Extract metadata
        if isinstance(checkpoint, dict):
            # Check for ultralytics YOLO signature
            if _is_ultralytics_yolo(checkpoint):
                info['backend'] = 'yolo'
                info['metadata']['framework'] = 'ultralytics'
                info['metadata']['task'] = guess_model_task(checkpoint.get('model', {}))
                
            # Check for RT-DETR signature
            elif _is_rtdetr(checkpoint):
                info['backend'] = 'rt-detr'
                info['metadata']['framework'] = 'ultralytics' if 'model' in checkpoint else 'custom'
                
            # Check for DETR signature
            elif _is_detr(checkpoint):
                info['backend'] = 'detr'
                info['metadata']['framework'] = 'transformers' if 'model' in checkpoint else 'custom'
                
            # Generic PyTorch model
            else:
                info['backend'] = _guess_from_architecture(checkpoint)
                info['metadata']['keys'] = list(checkpoint.keys())
        
        # Model is a direct nn.Module (less common)
        else:
            info['backend'] = _guess_from_module(checkpoint)
        
        return info
        
    except Exception as e:
        logger.warning(f"Failed to inspect PyTorch model: {e}")
        return {
            'backend': None,
            'format': 'pytorch',
            'error': str(e)
        }

def _is_ultralytics_yolo(checkpoint: dict) -> bool:
    """Check if model is ultralytics YOLO."""
    # Ultralytics YOLO models have specific keys
    yolo_keys = ['model', 'epoch', 'best_fitness', 'optimizer', 'train_args']
    
    # Check for YOLO signature
    if 'model' in checkpoint:
        model = checkpoint.get('model')
        if hasattr(model, 'yaml') or isinstance(model, dict) and 'yaml' in model:
            return True
    
    # Check for training metadata
    if 'train_args' in checkpoint or 'ema' in checkpoint:
        return True
    
    # Check for YOLO-specific keys
    has_yolo_keys = sum(1 for key in yolo_keys if key in checkpoint)
    return has_yolo_keys >= 2


def _is_rtdetr(checkpoint: dict) -> bool:
    """Check if model is RT-DETR."""
    # RT-DETR specific indicators
    if 'model' in checkpoint:
        model = checkpoint.get('model')
        
        # Check model name/yaml
        if hasattr(model, 'names') or isinstance(model, dict):
            yaml_content = str(checkpoint.get('model', {}).get('yaml', ''))
            if 'rtdetr' in yaml_content.lower():
                return True
    
    # Check state dict keys for RT-DETR patterns
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        keys_str = ' '.join(str(k) for k in list(state_dict.keys())[:20])
        if 'detr' in keys_str.lower() and ('encoder' in keys_str or 'decoder' in keys_str):
            return True
    
    return False


def _is_detr(checkpoint: dict) -> bool:
    """Check if model is DETR (Facebook/HuggingFace)."""
    # HuggingFace DETR signature
    if 'model' in checkpoint:
        state_dict_keys = checkpoint.get('model', {}).keys() if isinstance(checkpoint.get('model'), dict) else []
        keys_str = ' '.join(str(k) for k in list(state_dict_keys)[:20])
        
        # DETR has specific architecture patterns
        if 'detr' in keys_str.lower() and 'class_embed' in keys_str:
            return True
    
    # Check for config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_type = config.get('model_type', '').lower()
            if 'detr' in model_type and 'rt' not in model_type:
                return True
    
    return False


def _guess_from_architecture(checkpoint: dict) -> Optional[str]:
    """
    Guess backend from model architecture keys.
    
    This is a fallback when specific signatures aren't found.
    """
    # Get all keys as string
    if 'state_dict' in checkpoint:
        keys = list(checkpoint['state_dict'].keys())
    elif 'model' in checkpoint:
        if isinstance(checkpoint['model'], dict):
            keys = list(checkpoint['model'].keys())
        else:
            keys = []
    else:
        keys = list(checkpoint.keys())
    
    keys_str = ' '.join(str(k) for k in keys[:50]).lower()
    
    # Look for architecture patterns
    if 'backbone' in keys_str and 'neck' in keys_str and 'head' in keys_str:
        return 'yolo'  # YOLO-like architecture
    
    if 'transformer' in keys_str and 'query' in keys_str:
        return 'detr'  # Transformer-based detector
    
    if 'efficientnet' in keys_str or 'bifpn' in keys_str:
        return 'efficientdet'
    
    return None


def _guess_from_module(module) -> Optional[str]:
    """Guess backend from PyTorch module."""
    module_str = str(type(module)).lower()
    
    if 'yolo' in module_str:
        return 'yolo'
    if 'detr' in module_str:
        if 'rt' in module_str:
            return 'rt-detr'
        return 'detr'
    
    return None


def _inspect_onnx_model(model_path: Path) -> Dict[str, Any]:
    """Inspect ONNX model to determine backend."""
    try:
        import onnx
        
        model = onnx.load(str(model_path))
        
        info = {
            'format': 'onnx',
            'backend': None,
            'metadata': {}
        }
        
        # Extract metadata
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        info['metadata'] = metadata
        
        # Try to determine backend from metadata
        if 'description' in metadata:
            desc = metadata['description'].lower()
            if 'yolo' in desc:
                info['backend'] = 'yolo'
            elif 'detr' in desc:
                info['backend'] = 'detr'
        
        # Check producer name
        producer = model.producer_name.lower()
        if 'ultralytics' in producer:
            info['backend'] = 'yolo'
        
        return info
        
    except Exception as e:
        logger.warning(f"Failed to inspect ONNX model: {e}")
        return {
            'backend': None,
            'format': 'onnx',
            'error': str(e)
        }


def detect_backend(model_path: Path, hint: Optional[str] = None) -> str:
    """
    Detect the appropriate backend for a model.
    
    Args:
        model_path: Path to model file
        hint: Optional hint (e.g., from user specification)
        
    Returns:
        Backend name
        
    Raises:
        ValueError: If backend cannot be determined
    """
    # If hint provided and valid, use it
    if hint:
        from .registry import DetectorRegistry
        if hint in DetectorRegistry.list_backends():
            logger.info(f"Using specified backend: {hint}")
            return hint
        else:
            logger.warning(f"Invalid backend hint '{hint}', will auto-detect")
    
    # Inspect model
    info = inspect_model(model_path)
    
    if info.get('backend'):
        logger.info(f"Detected backend: {info['backend']} (confidence: high)")
        return info['backend']
    
    # Fallback to extension-based detection
    from .registry import DetectorRegistry
    backend = DetectorRegistry.detect_backend(model_path)
    
    if backend:
        logger.warning(
            f"Detected backend: {backend} (confidence: low, based on extension only). "
            f"Consider specifying backend explicitly."
        )
        return backend
    
    raise ValueError(
        f"Could not detect backend for {model_path}. "
        f"Please specify backend explicitly using backend='backend_name'"
    )