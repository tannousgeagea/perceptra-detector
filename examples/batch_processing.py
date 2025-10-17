"""
Batch processing example with progress tracking.
"""

from pathlib import Path
from tqdm import tqdm

from perceptra_detector import Detector
from perceptra_detector.utils.visualization import draw_detections
from perceptra_detector.utils.image import load_image, save_image


def main():
    # Configuration
    model_path = "yolov8n.pt"
    input_dir = Path("input_images")
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    print("Loading model...")
    detector = Detector(
        model_path=model_path,
        confidence_threshold=0.5,
        auto_warmup=True  # Warmup for better performance
    )
    
    # Find all images
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_paths.extend(input_dir.glob(f'*{ext}'))
    
    print(f"\nFound {len(image_paths)} images")
    
    # Method 1: Process one by one with custom progress
    print("\nMethod 1: Sequential processing with progress bar")
    all_results = []
    
    for img_path in tqdm(image_paths, desc="Processing"):
        # Load and detect
        image = load_image(img_path)
        result = detector.detect(image)
        all_results.append(result)
        
        # Save annotated image
        annotated = draw_detections(image, result)
        output_path = output_dir / img_path.name
        save_image(annotated, output_path)
    
    # Method 2: Batch processing (more efficient for some backends)
    print("\nMethod 2: Batch processing")
    batch_result = detector.detect_batch(
        [load_image(p) for p in image_paths],
        confidence_threshold=0.5
    )
    
    print(f"\nBatch processing stats:")
    print(f"  Total images: {len(batch_result)}")
    print(f"  Total time: {batch_result.total_inference_time:.2f}s")
    print(f"  Average time: {batch_result.total_inference_time/len(batch_result)*1000:.1f}ms per image")
    
    # Aggregate statistics
    total_detections = sum(len(r) for r in batch_result.results)
    class_counts = {}
    
    for result in batch_result.results:
        for class_name, count in result.get_class_counts().items():
            class_counts[class_name] = class_counts.get(class_name, 0) + count
    
    print(f"\nAggregate statistics:")
    print(f"  Total detections: {total_detections}")
    print(f"  Unique classes: {len(class_counts)}")
    print(f"\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    # Method 3: Directory processing (built-in)
    print("\nMethod 3: Directory processing (built-in)")
    batch_result = detector.detect_directory(
        directory=input_dir,
        recursive=False,
        confidence_threshold=0.5
    )
    
    print(f"Processed {len(batch_result)} images from directory")


if __name__ == "__main__":
    main()