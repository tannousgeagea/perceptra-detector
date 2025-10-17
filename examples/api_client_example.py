"""
Example using the Perceptra Detector API client.
"""

from perceptra_detector.client import DetectorClient
import json


def main():
    # Initialize client
    client = DetectorClient(
        base_url="http://localhost:8000",
        timeout=30
    )
    
    print("="*50)
    print("PERCEPTRA DETECTOR API CLIENT")
    print("="*50)
    
    # 1. Health check
    print("\n1. Health Check")
    print("-" * 30)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Models loaded: {health['models_loaded']}")
        print(f"GPU available: {health['gpu_available']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 2. List models
    print("\n2. Available Models")
    print("-" * 30)
    models_info = client.list_models()
    print(f"Default model: {models_info['default_model']}")
    print(f"Available models: {', '.join(models_info['models'])}")
    
    # 3. Get model information
    print("\n3. Model Information")
    print("-" * 30)
    for model_name in models_info['models']:
        info = client.get_model_info(model_name)
        print(f"\nModel: {model_name}")
        print(f"  Backend: {info['backend']}")
        print(f"  Device: {info['device']}")
        print(f"  Task: {info['task_type']}")
        print(f"  Classes: {info['num_classes']}")
    
    # 4. Single image detection
    print("\n4. Single Image Detection")
    print("-" * 30)
    try:
        result = client.detect(
            image="test_image.jpg",
            confidence_threshold=0.5
        )
        
        print(f"Detections: {result['num_detections']}")
        print(f"Inference time: {result['inference_time']*1000:.1f}ms")
        print(f"Image size: {result['image_shape']['width']}x{result['image_shape']['height']}")
        
        print("\nDetected objects:")
        for det in result['detections']:
            print(f"  - {det['class_name']}: {det['confidence']:.3f}")
        
        print("\nClass distribution:")
        for class_name, count in result['class_counts'].items():
            print(f"  - {class_name}: {count}")
        
    except FileNotFoundError:
        print("test_image.jpg not found - skipping single detection")
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Batch detection
    print("\n5. Batch Detection")
    print("-" * 30)
    try:
        images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        batch_result = client.detect_batch(
            images=images,
            confidence_threshold=0.5
        )
        
        print(f"Processed: {batch_result['num_images']} images")
        print(f"Total time: {batch_result['total_inference_time']:.2f}s")
        print(f"Average time: {batch_result['average_inference_time']*1000:.1f}ms per image")
        
        for idx, result in enumerate(batch_result['results']):
            print(f"\nImage {idx+1}: {result['num_detections']} detections")
            
    except Exception as e:
        print(f"Batch detection skipped: {e}")
    
    # 6. URL detection
    print("\n6. URL Detection")
    print("-" * 30)
    try:
        url = "https://ultralytics.com/images/bus.jpg"
        result = client.detect_url(
            url=url,
            confidence_threshold=0.5
        )
        
        print(f"URL: {url}")
        print(f"Detections: {result['num_detections']}")
        
        for det in result['detections'][:5]:  # Show first 5
            print(f"  - {det['class_name']}: {det['confidence']:.3f}")
        
    except Exception as e:
        print(f"URL detection failed: {e}")
    
    # 7. Save results
    print("\n7. Saving Results")
    print("-" * 30)
    try:
        result = client.detect("test_image.jpg")
        
        # Save as JSON
        with open("api_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("✓ Saved results to api_results.json")
        
    except Exception as e:
        print(f"Could not save results: {e}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


def streaming_example():
    """Example: Process multiple images with progress tracking."""
    from pathlib import Path
    from tqdm import tqdm
    
    client = DetectorClient("http://localhost:8000")
    
    # Find all images
    image_dir = Path("images")
    image_paths = list(image_dir.glob("*.jpg"))
    
    print(f"Processing {len(image_paths)} images...")
    
    results = []
    for img_path in tqdm(image_paths):
        try:
            result = client.detect(str(img_path))
            results.append({
                'filename': img_path.name,
                'detections': result['num_detections'],
                'inference_time': result['inference_time']
            })
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")
    
    # Summary
    total_detections = sum(r['detections'] for r in results)
    avg_time = sum(r['inference_time'] for r in results) / len(results)
    
    print(f"\nProcessed {len(results)} images")
    print(f"Total detections: {total_detections}")
    print(f"Average time: {avg_time*1000:.1f}ms")


def async_example():
    """Example: Async processing with aiohttp (optional)."""
    import asyncio
    import aiohttp
    
    async def detect_async(session, url, image_path):
        """Async detection request."""
        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=image_path.name)
            
            async with session.post(f"{url}/detect", data=data) as response:
                return await response.json()
    
    async def process_batch_async(image_paths):
        """Process multiple images concurrently."""
        base_url = "http://localhost:8000"
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                detect_async(session, base_url, img_path)
                for img_path in image_paths
            ]
            results = await asyncio.gather(*tasks)
            return results
    
    # Run
    from pathlib import Path
    image_paths = list(Path("images").glob("*.jpg"))[:10]
    
    print("Async processing...")
    results = asyncio.run(process_batch_async(image_paths))
    print(f"Processed {len(results)} images concurrently")


if __name__ == "__main__":
    main()
    
    # Uncomment to try other examples:
    # streaming_example()
    # async_example()  # Requires: pip install aiohttp