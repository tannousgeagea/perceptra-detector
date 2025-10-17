"""
Video processing example with real-time statistics.
"""

from perceptra_detector import Detector
from collections import defaultdict


def frame_callback(frame_idx, result):
    """Callback function called for each processed frame."""
    if frame_idx % 30 == 0:  # Print every 30 frames
        print(f"Frame {frame_idx}: {len(result)} detections, "
              f"{result.inference_time*1000:.1f}ms")


def main():
    # Initialize detector
    print("Loading model...")
    detector = Detector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5,
        device="cuda",
        auto_warmup=True
    )
    
    # Process video
    print("\nProcessing video...")
    results = detector.detect_video(
        video_path="input_video.mp4",
        output_path="output_video.mp4",
        skip_frames=0,  # Process all frames (set to N to process every N+1 frames)
        show_progress=True
    )
    
    # Analyze results
    print("\n" + "="*50)
    print("VIDEO ANALYSIS")
    print("="*50)
    
    total_frames = len(results)
    total_detections = sum(len(r) for r in results)
    avg_inference_time = sum(r.inference_time for r in results) / total_frames
    
    print(f"\nOverall Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections/total_frames:.2f}")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms")
    print(f"  Effective FPS: {1/avg_inference_time:.1f}")
    
    # Track objects across frames
    class_history = defaultdict(list)
    for frame_idx, result in enumerate(results):
        counts = result.get_class_counts()
        for class_name in detector.class_names:
            class_history[class_name].append(counts.get(class_name, 0))
    
    # Print temporal statistics
    print(f"\nTemporal Statistics:")
    for class_name, counts in class_history.items():
        total = sum(counts)
        if total > 0:
            max_count = max(counts)
            avg_count = total / total_frames
            presence = sum(1 for c in counts if c > 0)
            print(f"  {class_name}:")
            print(f"    Total appearances: {total}")
            print(f"    Max per frame: {max_count}")
            print(f"    Average per frame: {avg_count:.2f}")
            print(f"    Present in: {presence}/{total_frames} frames ({presence/total_frames*100:.1f}%)")
    
    # Find interesting frames (frames with most detections)
    print(f"\nTop 5 frames by detection count:")
    frame_scores = [(i, len(r)) for i, r in enumerate(results)]
    top_frames = sorted(frame_scores, key=lambda x: x[1], reverse=True)[:5]
    
    for rank, (frame_idx, count) in enumerate(top_frames, 1):
        print(f"  {rank}. Frame {frame_idx}: {count} detections")
        result = results[frame_idx]
        class_counts = result.get_class_counts()
        print(f"     Classes: {', '.join(f'{k}({v})' for k, v in class_counts.items())}")


def process_video_stream():
    """Example: Process video with custom frame-by-frame logic."""
    import cv2
    from perceptra_detector.utils.visualization import draw_detections
    
    detector = Detector("yolov8n.pt", auto_warmup=True)
    
    cap = cv2.VideoCapture("input_video.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('custom_output.mp4', fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        result = detector.detect(frame_rgb)
        
        # Custom processing based on detections
        if len(result) > 0:
            # Draw detections
            annotated = draw_detections(frame_rgb, result)
            frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Add custom text/overlays
            text = f"Frame {frame_idx}: {len(result)} objects"
            cv2.putText(frame_bgr, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame_bgr = frame
        
        out.write(frame_bgr)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Processed {frame_idx} frames")


if __name__ == "__main__":
    main()
    # Uncomment to try custom processing:
    # process_video_stream()