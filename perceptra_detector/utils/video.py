"""
Video processing utilities.
"""

from typing import Optional, List, Union, Callable
from pathlib import Path
import logging

from ..core.base import BaseDetector
from ..core.schemas import DetectionResult
from .visualization import draw_detections

logger = logging.getLogger(__name__)


def process_video(
    video_path: Union[str, Path],
    detector: BaseDetector,
    output_path: Optional[Union[str, Path]] = None,
    confidence_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
    visualize: bool = True,
    callback: Optional[Callable[[int, DetectionResult], None]] = None
) -> List[DetectionResult]:
    """
    Process video file with detector.
    
    Args:
        video_path: Path to input video
        detector: Detector instance
        output_path: Optional path to save annotated video
        confidence_threshold: Override confidence threshold
        iou_threshold: Override IoU threshold
        skip_frames: Process every nth frame
        max_frames: Maximum frames to process
        show_progress: Show progress bar
        visualize: Draw detections on output video
        callback: Optional callback(frame_idx, result)
        
    Returns:
        List of detection results
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(
        f"Video: {width}x{height} @ {fps}fps, "
        f"total_frames={total_frames}"
    )
    
    # Setup video writer if output requested
    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
    
    # Setup progress bar
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames)
        except ImportError:
            logger.warning("tqdm not available, progress bar disabled")
    
    results = []
    frame_idx = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check max frames
            if max_frames and processed_frames >= max_frames:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                if pbar:
                    pbar.update(1)
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            result = detector.detect(
                frame_rgb,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            results.append(result)
            
            # Call callback if provided
            if callback:
                callback(frame_idx, result)
            
            # Draw detections and save if output requested
            if writer:
                if visualize:
                    annotated = draw_detections(frame_rgb, result)
                    frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
            
            frame_idx += 1
            processed_frames += 1
            
            if pbar:
                pbar.update(1)
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()
    
    logger.info(f"Processed {processed_frames} frames")
    
    if output_path:
        logger.info(f"Saved annotated video to {output_path}")
    
    return results


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: Optional[int] = None,
    max_frames: Optional[int] = None,
    prefix: str = "frame"
) -> List[Path]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Extract at this FPS (None = all frames)
        max_frames: Maximum frames to extract
        prefix: Filename prefix
        
    Returns:
        List of extracted frame paths
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required")
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame skip
    skip = 1
    if fps is not None and fps < video_fps:
        skip = int(video_fps / fps)
    
    frame_paths = []
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if max_frames and saved_count >= max_frames:
            break
        
        if frame_idx % skip == 0:
            frame_path = output_dir / f"{prefix}_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames to {output_dir}")
    
    return frame_paths


class VideoWriter:
    """
    Simple video writer wrapper.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 30,
        frame_size: Optional[tuple] = None,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video path
            fps: Frames per second
            frame_size: Frame size (width, height) - auto-detected from first frame if None
            codec: Video codec
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required")
        
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
        self.frame_count = 0
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, frame):
        """Write a frame to video."""
        import cv2
        
        if self.writer is None:
            # Initialize writer on first frame
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size
            )
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()
            logger.info(
                f"Saved {self.frame_count} frames to {self.output_path}"
            )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()