"""
Command-line interface for perceptra-detector.
"""

import click
import sys
from pathlib import Path
import logging

from ..core.detector import Detector
from ..utils.visualization import draw_detections
from ..utils.image import save_image
from ..__version__ import __version__

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def cli():
    """Perceptra Detector CLI - Production-ready object detection."""
    pass


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output image path')
@click.option('--confidence', '-c', type=float, default=0.25, help='Confidence threshold')
@click.option('--iou', type=float, default=0.45, help='IoU threshold')
@click.option('--backend', '-b', type=str, help='Backend to use (auto-detected if not specified)')
@click.option('--device', type=str, help='Device (cuda/cpu)')
@click.option('--visualize/--no-visualize', default=True, help='Draw detections on image')
@click.option('--save-json', type=click.Path(), help='Save results as JSON')
def detect(model_path, image_path, output, confidence, iou, backend, device, visualize, save_json):
    """
    Run detection on a single image.
    
    Example:
        perceptra-detector detect yolov8n.pt image.jpg -o output.jpg
    """
    try:
        # Load detector
        click.echo(f"Loading model: {model_path}")
        detector = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            confidence_threshold=confidence,
            iou_threshold=iou
        )
        
        # Run detection
        click.echo(f"Processing: {image_path}")
        result = detector.detect(image_path)
        
        # Print results
        click.echo(f"\n✓ Found {len(result)} detections")
        click.echo(f"  Inference time: {result.inference_time*1000:.1f}ms")
        
        # Print class counts
        class_counts = result.get_class_counts()
        if class_counts:
            click.echo("\nDetected classes:")
            for class_name, count in sorted(class_counts.items()):
                click.echo(f"  - {class_name}: {count}")
        
        # Save visualization
        if output and visualize:
            from ..utils.image import load_image
            image = load_image(image_path)
            annotated = draw_detections(image, result)
            save_image(annotated, output)
            click.echo(f"\n✓ Saved visualization to: {output}")
        
        # Save JSON
        if save_json:
            import json
            with open(save_json, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            click.echo(f"✓ Saved results to: {save_json}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--confidence', '-c', type=float, default=0.25)
@click.option('--iou', type=float, default=0.45)
@click.option('--backend', '-b', type=str)
@click.option('--device', type=str)
@click.option('--recursive', '-r', is_flag=True, help='Search subdirectories')
@click.option('--extensions', default='.jpg,.jpeg,.png,.bmp', help='Image extensions')
@click.option('--save-json', type=click.Path(), help='Save batch results as JSON')
def batch(model_path, input_dir, output_dir, confidence, iou, backend, device, recursive, extensions, save_json):
    """
    Run detection on all images in a directory.
    
    Example:
        perceptra-detector batch yolov8n.pt images/ -o results/
    """
    try:
        # Parse extensions
        ext_list = [e.strip() for e in extensions.split(',')]
        
        # Load detector
        click.echo(f"Loading model: {model_path}")
        detector = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            confidence_threshold=confidence,
            iou_threshold=iou
        )
        
        # Process directory
        click.echo(f"Processing directory: {input_dir}")
        batch_result = detector.detect_directory(
            directory=input_dir,
            extensions=ext_list,
            recursive=recursive
        )
        
        # Print summary
        click.echo(f"\n✓ Processed {len(batch_result)} images")
        click.echo(f"  Total time: {batch_result.total_inference_time:.2f}s")
        click.echo(f"  Average time: {batch_result.total_inference_time/len(batch_result)*1000:.1f}ms per image")
        
        # Save visualizations
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            from ..utils.image import load_image
            input_path = Path(input_dir)
            
            # Find all images
            image_paths = []
            for ext in ext_list:
                if recursive:
                    image_paths.extend(input_path.rglob(f"*{ext}"))
                else:
                    image_paths.extend(input_path.glob(f"*{ext}"))
            
            for img_path, result in zip(sorted(image_paths)[:len(batch_result)], batch_result.results):
                image = load_image(img_path)
                annotated = draw_detections(image, result)
                out_path = output_path / img_path.name
                save_image(annotated, out_path)
            
            click.echo(f"\n✓ Saved visualizations to: {output_dir}")
        
        # Save JSON
        if save_json:
            import json
            with open(save_json, 'w') as f:
                json.dump(batch_result.to_dict(), f, indent=2)
            click.echo(f"✓ Saved results to: {save_json}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output video path')
@click.option('--confidence', '-c', type=float, default=0.25)
@click.option('--iou', type=float, default=0.45)
@click.option('--backend', '-b', type=str)
@click.option('--device', type=str)
@click.option('--skip-frames', type=int, default=0, help='Process every nth frame')
@click.option('--max-frames', type=int, help='Maximum frames to process')
def video(model_path, video_path, output, confidence, iou, backend, device, skip_frames, max_frames):
    """
    Run detection on video frames.
    
    Example:
        perceptra-detector video yolov8n.pt input.mp4 -o output.mp4
    """
    try:
        # Load detector
        click.echo(f"Loading model: {model_path}")
        detector = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            confidence_threshold=confidence,
            iou_threshold=iou
        )
        
        # Process video
        click.echo(f"Processing video: {video_path}")
        results = detector.detect_video(
            video_path=video_path,
            output_path=output,
            skip_frames=skip_frames,
            max_frames=max_frames,
            show_progress=True
        )
        
        # Print summary
        total_detections = sum(len(r) for r in results)
        avg_time = sum(r.inference_time for r in results) / len(results)
        
        click.echo(f"\n✓ Processed {len(results)} frames")
        click.echo(f"  Total detections: {total_detections}")
        click.echo(f"  Average inference time: {avg_time*1000:.1f}ms")
        
        if output:
            click.echo(f"✓ Saved annotated video to: {output}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind')
@click.option('--port', default=8000, type=int, help='Port to bind')
@click.option('--models', '-m', multiple=True, help='Models to load (format: name:path)')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', type=int, default=1, help='Number of workers')
def serve(host, port, models, reload, workers):
    """
    Start the detection API server.
    
    Example:
        perceptra-detector serve -m yolo:yolov8n.pt -m detr:detr-model.pth
    """
    try:
        import uvicorn
        from ..api.server import create_app
        
        # Parse models
        models_config = {}
        for model_spec in models:
            parts = model_spec.split(':', 2)
            
            if len(parts) == 2:
                # Format: name:path (auto-detect backend)
                name, path = parts
                models_config[name] = path
            elif len(parts) == 3:
                # Format: name:backend:path (explicit backend)
                name, backend, path = parts
                models_config[name] = {'path': path, 'backend': backend}
            else:
                click.echo(f"✗ Invalid model specification: {model_spec}", err=True)
                click.echo("  Format: name:path or name:backend:path", err=True)
                sys.exit(1)
        
        if not models_config:
            click.echo("⚠ No models specified. Server will start but no models will be available.")
            click.echo("  Use -m name:path to load models")
        
        # Create app
        app = create_app(models_config=models_config)
        
        # Start server
        click.echo(f"Starting server on {host}:{port}")
        if models_config:
            click.echo(f"Loaded models: {', '.join(models_config.keys())}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers
        )
    
    except ImportError as e:
        click.echo(f"✗ Missing dependency: {e}", err=True)
        click.echo("  Install with: pip install uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_backends():
    """List available detection backends."""
    backends = Detector.list_backends()
    extensions = Detector.list_supported_extensions()
    
    click.echo("Available backends:\n")
    for backend in backends:
        exts = [ext for ext, b in extensions.items() if b == backend]
        click.echo(f"  • {backend}")
        if exts:
            click.echo(f"    Extensions: {', '.join(exts)}")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()