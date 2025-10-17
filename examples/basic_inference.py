"""
Basic inference example with Perceptra Detector.
"""

from perceptra_detector import Detector
from perceptra_detector.utils.visualization import draw_detections
from perceptra_detector.utils.image import load_image, save_image


def main():
    # Initialize detector
    print("Loading model...")
    detector = Detector(
        model_path="yolov8n.pt",  # Can be any supported model
        confidence_threshold=0.25,
        device="cuda"  # or "cpu"
    )
    
    # Load image
    print("Loading image...")
    image = load_image("path/to/image.jpg")
    
    # Run detection
    print("Running detection...")
    result = detector.detect(image)
    
    # Print results
    print(f"\nFound {len(result)} detections")
    print(f"Inference time: {result.inference_time*1000:.1f}ms")
    
    # Print each detection
    for i, detection in enumerate(result.detections, 1):
        print(f"\nDetection {i}:")
        print(f"  Class: {detection.class_name}")
        print(f"  Confidence: {detection.confidence:.3f}")
        print(f"  BBox: ({detection.bbox.x1:.0f}, {detection.bbox.y1:.0f}, "
              f"{detection.bbox.x2:.0f}, {detection.bbox.y2:.0f})")
    
    # Print class distribution
    print("\nClass distribution:")
    for class_name, count in result.get_class_counts().items():
        print(f"  {class_name}: {count}")
    
    # Visualize and save
    print("\nDrawing detections...")
    annotated = draw_detections(image, result)
    save_image(annotated, "output.jpg")
    print("Saved to output.jpg")
    
    # Export to JSON
    print("\nExporting to JSON...")
    with open("results.json", "w") as f:
        f.write(result.to_json())
    print("Saved to results.json")


if __name__ == "__main__":
    main()