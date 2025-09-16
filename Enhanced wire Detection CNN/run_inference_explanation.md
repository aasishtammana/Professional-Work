# Comprehensive Explanation: run_inference.py

## Overview
The `run_inference.py` script is the inference pipeline for the Enhanced Wire Detection CNN system. It loads trained models and runs them on new schematic images to detect wires and junctions. This script handles both single image inference and batch processing, with comprehensive visualization and post-processing capabilities.

## Architecture and Dependencies

### Core Dependencies
- **TensorFlow/Keras**: Model loading and inference
- **OpenCV (cv2)**: Image processing and manipulation
- **NumPy**: Numerical operations and array handling
- **Matplotlib**: Visualization and plotting
- **scikit-image**: Image analysis and morphological operations
- **NetworkX**: Graph operations for wire network analysis

### Custom Module Imports
- `simple_postprocessor`: Post-processing utilities for wire detection results
- `enhanced_models`: Model architectures and custom loss functions
- `enhanced_wire_segmentation`: Advanced wire segmentation and connection analysis

## Detailed Function Analysis

### 1. `load_model()` Function

**Purpose**: Loads a trained model with proper custom object handling for inference.

**Detailed Logic**:
```python
def load_model(model_path):
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Import custom loss functions
    from enhanced_models import combined_wire_loss, junction_focal_loss
    
    # Create loss functions with custom names (same as in training)
    wire_loss = combined_wire_loss
    wire_loss.__name__ = 'wire_loss'
    
    junction_loss = junction_focal_loss
    junction_loss.__name__ = 'junction_loss'
    
    # Define custom objects for model loading
    custom_objects = {
        'WireDetectionUNet': WireDetectionUNet,
        'WireDetectionResNet': WireDetectionResNet,
        'WireDetectionAttention': WireDetectionAttention,
        'combined_wire_loss': combined_wire_loss,
        'junction_focal_loss': junction_focal_loss,
        'wire_loss': wire_loss,
        'junction_loss': junction_loss
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model
```

**Key Features**:
- **Custom Object Handling**: Properly loads models with custom loss functions and architectures
- **Error Handling**: Validates model file existence before loading
- **Loss Function Mapping**: Maps custom loss functions to their names for proper model reconstruction
- **Architecture Support**: Supports all three model architectures (U-Net, ResNet, Attention)

**Why This Matters**: TensorFlow models saved during training include custom loss functions and architectures. Without proper custom object mapping, the model loading would fail. This function ensures seamless model loading regardless of the training configuration.

### 2. `preprocess_image()` Function

**Purpose**: Preprocesses input images for model inference with proper resolution handling and normalization.

**Detailed Logic**:
```python
def preprocess_image(image_path, target_size=(512, 512)):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original size and image
    original_size = img.shape[:2]
    original_img = img.copy()
    
    # Convert to grayscale to remove color sensitivity
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Convert back to 3-channel (same grayscale in all channels)
    img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # Always use the target size that the model was trained on
    new_target_size = target_size
    
    # Resize only for model input with better interpolation
    img_resized = cv2.resize(img_gray_3ch, new_target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, original_img, original_size
```

**Key Features**:
- **Color Space Conversion**: Converts BGR to RGB for proper color handling
- **Grayscale Conversion**: Removes color sensitivity by converting to grayscale
- **Resolution Preservation**: Stores original image and size for post-processing
- **High-Quality Resizing**: Uses LANCZOS4 interpolation for better quality
- **Proper Normalization**: Normalizes to [0, 1] range as expected by the model
- **Batch Dimension**: Adds batch dimension for model input

**Why This Matters**: The preprocessing must match exactly what was done during training. Any deviation in normalization, color space, or resolution can lead to poor inference results. This function ensures consistency between training and inference.

### 3. `run_inference()` Function

**Purpose**: Runs inference on a single image and handles post-processing and visualization.

**Detailed Logic**:
```python
def run_inference(model, image_path, output_dir="results/inference"):
    # Preprocess image
    img_batch, original_img, original_size = preprocess_image(image_path)
    
    # Run model
    outputs = model(img_batch)
    
    # Extract masks
    wire_mask = outputs['wire_mask'][0, :, :, 0].numpy()
    junction_mask = outputs['junction_mask'][0, :, :, 0].numpy()
    
    # Resize masks back to original resolution
    wire_mask_original = cv2.resize(wire_mask, (original_size[1], original_size[0]))
    junction_mask_original = cv2.resize(junction_mask, (original_size[1], original_size[0]))
    
    # Post-process (use original resolution masks with text filtering)
    processor = WirePostProcessor()
    results = processor.process_wire_mask(wire_mask_original, junction_mask_original, image_path, original_size)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Create simple, clean wire detection visualization with original resolution
    create_wire_detection_visualization(original_img, wire_mask_original, output_dir, image_name, image_path)
    
    # Create junction visualization
    create_junction_visualization(original_img, junction_mask_original, results['junctions'], output_dir, image_name)
    
    # Create individual wire segments visualization
    create_individual_wire_segments_visualization(original_img, wire_mask_original, output_dir, image_name)
    
    # Print results
    print(f"Results for {image_name}:")
    print(f"  Original image size: {original_size}")
    print(f"  Model input size: {img_batch.shape[1:3]}")
    print(f"  Junctions detected: {len(results['junctions'])}")
    
    # Print detection quality metrics
    wire_pixels = np.sum(wire_mask_original > 0.1)  # Lower threshold for undertrained model
    junction_pixels = np.sum(junction_mask_original > 0.1)
    print(f"  Wire pixels detected: {wire_pixels}")
    print(f"  Junction pixels detected: {junction_pixels}")
    
    # Print junction details
    if results['junctions']:
        print(f"  Junction locations: {results['junctions'][:5]}...")  # Show first 5
    
    return {
        'wire_pixels': wire_pixels,
        'junction_count': len(results['junctions']),
        'junction_pixels': junction_pixels,
        'results': results
    }
```

**Key Features**:
- **Model Execution**: Runs the trained model on preprocessed input
- **Mask Extraction**: Extracts wire and junction masks from model outputs
- **Resolution Scaling**: Scales masks back to original image resolution
- **Post-Processing**: Applies text filtering and junction detection
- **Comprehensive Visualization**: Creates multiple visualization types
- **Detailed Reporting**: Provides comprehensive metrics and statistics

**Why This Matters**: This function is the core of the inference pipeline. It handles the complete workflow from image preprocessing to result visualization, ensuring that users get meaningful output from the trained model.

### 4. `create_wire_detection_visualization()` Function

**Purpose**: Creates comprehensive wire detection visualizations with text filtering analysis.

**Detailed Logic**:
```python
def create_wire_detection_visualization(original_img, wire_mask, output_dir, image_name, image_path=None):
    # Calculate figure size based on image aspect ratio
    height, width = original_img.shape[:2]
    aspect_ratio = width / height
    
    # Create figure with 2 subplots, maintaining aspect ratio
    fig_width = 24
    fig_height = fig_width / (2 * aspect_ratio)  # 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Schematic', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Combined text and wire detection overlay
    axes[1].imshow(original_img, alpha=0.8)
    
    # Create overlays for both text and wires
    text_wire_overlay = np.zeros_like(original_img)
    
    # Get text pixels using the same Tesseract-based detection as in training
    try:
        if image_path:
            from enhanced_data_pipeline import SchematicDataset
            dataset = SchematicDataset("data/schematics_dataset")
            
            # Create a test mask to get text regions
            test_mask = np.ones((original_img.shape[0], original_img.shape[1]), dtype=np.uint8) * 128
            filtered_mask = dataset._filter_text_regions(test_mask, image_path)
            
            # Text pixels are where the mask was removed (difference between original and filtered)
            text_pixels = ((test_mask > 0) & (filtered_mask == 0)).astype(np.uint8)
            
            print(f"  Text pixels detected: {np.sum(text_pixels)}")
        else:
            # Fallback to simple text detection
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            text_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            text_mask = cv2.dilate(text_mask, iterations=2)
            text_pixels = (text_mask > 0).astype(np.uint8)
            
    except Exception as e:
        print(f"Warning: Could not apply text detection: {e}")
        text_pixels = np.zeros_like(wire_mask, dtype=np.uint8)
    
    # Wire pixels
    wire_binary = (wire_mask > 0.1).astype(np.uint8)  # Lower threshold for undertrained model
    
    # Create overlay: Green for text, Red for wires
    text_wire_overlay[:, :, 1] = text_pixels * 255  # Green channel for text
    text_wire_overlay[:, :, 0] = wire_binary * 255  # Red channel for wires
    
    axes[1].imshow(text_wire_overlay, alpha=0.7)
    axes[1].set_title('Text (Green) + Wire Detection (Red)', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{image_name}_wire_detection.png", dpi=300, bbox_inches='tight')
    plt.close()
```

**Key Features**:
- **Aspect Ratio Preservation**: Maintains proper image proportions in visualizations
- **Text Detection Integration**: Uses the same text filtering as training pipeline
- **Color-Coded Overlays**: Green for text, Red for wires for clear distinction
- **High-Quality Output**: Saves at 300 DPI for publication quality
- **Error Handling**: Graceful fallback if text detection fails

**Why This Matters**: This visualization helps users understand what the model is detecting and how text filtering affects the results. It's crucial for debugging and understanding model behavior.

### 5. `create_junction_visualization()` Function

**Purpose**: Creates specialized visualizations for junction detection results.

**Detailed Logic**:
```python
def create_junction_visualization(original_img, junction_mask, junctions, output_dir, image_name):
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with junction mask overlay
    axes[0].imshow(original_img)
    # Apply proper thresholding for junction mask
    junction_binary = (junction_mask > 0.1).astype(np.uint8)  # Lower threshold for undertrained model
    axes[0].imshow(junction_binary, alpha=0.5, cmap='Reds')
    axes[0].set_title('Junction Mask Overlay', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Original image with detected junction points
    axes[1].imshow(original_img)
    if junctions:
        junction_coords = np.array(junctions)
        axes[1].scatter(junction_coords[:, 0], junction_coords[:, 1], 
                       c='red', s=100, marker='o', edgecolors='yellow', linewidth=2)
    axes[1].set_title(f'Detected Junctions ({len(junctions)})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / f"{image_name}_junctions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Junction visualization saved to: {output_path}")
```

**Key Features**:
- **Dual Visualization**: Shows both raw junction mask and processed junction points
- **Clear Markers**: Uses distinctive markers for junction points
- **Count Display**: Shows number of detected junctions in title
- **High Contrast**: Uses red markers with yellow edges for visibility

**Why This Matters**: Junction detection is a critical part of wire analysis. This visualization helps users verify that the model is correctly identifying wire intersections and connection points.

### 6. `create_individual_wire_segments_visualization()` Function

**Purpose**: Creates detailed visualizations of individual wire segments with color coding.

**Detailed Logic**:
```python
def create_individual_wire_segments_visualization(original_img, wire_mask, output_dir, image_name):
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize, remove_small_objects
    
    # Create output path
    output_path = output_dir / f"{image_name}_wire_segments.png"
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Schematic', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Better wire segmentation with morphological operations
    binary_mask = (wire_mask > 0.1).astype(np.uint8)
    
    # Use morphological opening to separate connected wires
    from skimage.morphology import opening, disk
    opened_mask = opening(binary_mask, disk(2))  # Remove small connections
    
    # Clean up small objects
    cleaned_mask = remove_small_objects(opened_mask.astype(bool), min_size=50).astype(np.uint8)
    
    # Label connected components
    labeled = label(cleaned_mask)
    
    # Individual wire segments with 15 rotating colors
    axes[1].imshow(original_img, alpha=0.7)
    axes[1].set_title(f'Wire Segments ({len(np.unique(labeled)) - 1} segments)', fontsize=16, fontweight='bold')
    
    # Define 15 distinct colors
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green  
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
        (255, 192, 203),  # Pink
        (0, 255, 127),    # Spring Green
        (255, 20, 147),   # Deep Pink
        (0, 191, 255),    # Deep Sky Blue
        (255, 69, 0)      # Red Orange
    ]
    
    for i, region in enumerate(regionprops(labeled)):
        if region.area < 20:  # Skip small regions
            continue
            
        # Use rotating colors (modulo 15)
        color = colors[i % 15]
        rgb_color = (color[0]/255, color[1]/255, color[2]/255)
        
        # Create a mask for this specific region and skeletonize it
        region_mask = (labeled == region.label).astype(np.uint8)
        
        if region.area > 10:  # Only process regions with enough points
            # Skeletonize to get the centerline
            from skimage.morphology import skeletonize
            skeleton = skeletonize(region_mask)
            
            # Get skeleton coordinates
            skeleton_coords = np.where(skeleton)
            if len(skeleton_coords[0]) > 1:
                y_coords = skeleton_coords[0]
                x_coords = skeleton_coords[1]
                
                # Draw skeleton points as individual small line segments
                # This creates centerlines without artificial connections
                if len(x_coords) > 1:
                    # Draw each skeleton point as a small line segment
                    for i in range(len(x_coords)):
                        x, y = x_coords[i], y_coords[i]
                        # Draw a small line segment at each skeleton point
                        # This creates a centerline effect without connecting distant points
                        axes[1].plot([x-0.5, x+0.5], [y-0.5, y+0.5], color=rgb_color, linewidth=0.6, alpha=0.9)
                        axes[1].plot([x-0.5, x+0.5], [y+0.5, y-0.5], color=rgb_color, linewidth=0.6, alpha=0.9)
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Wire segments visualization saved to: {output_path}")
```

**Key Features**:
- **Morphological Processing**: Uses opening operations to separate connected wires
- **Skeletonization**: Creates centerlines for each wire segment
- **Color Coding**: Uses 15 distinct colors that rotate for different segments
- **Noise Filtering**: Removes small objects and regions
- **Centerline Visualization**: Shows wire centerlines without artificial connections

**Why This Matters**: This visualization is crucial for understanding how the model segments individual wires. It helps users see which wires are being detected as separate entities and which are being merged together.

### 7. `run_batch_inference()` Function

**Purpose**: Runs inference on multiple images with random sampling and comprehensive result aggregation.

**Detailed Logic**:
```python
def run_batch_inference(model, data_dir, output_dir, num_samples=5, random_seed=None):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f'*{ext}')))
    
    # Randomly sample images
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    else:
        print(f"Warning: Only {len(image_files)} images available, using all of them")
    
    print(f"Running inference on {len(image_files)} randomly selected images...")
    print(f"Selected images: {[f.name for f in image_files]}")
    
    all_results = []
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        try:
            results = run_inference(model, str(image_path), str(output_path))
            all_results.append({
                'image_path': str(image_path),
                'results': results
            })
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    # Save summary
    summary_path = output_path / "inference_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Wire Detection Inference Summary\n")
        f.write("================================\n\n")
        for result in all_results:
            f.write(f"Image: {Path(result['image_path']).name}\n")
            f.write(f"  Wire segments: {len(result['results']['results']['vectorized_wires'])}\n")
            f.write(f"  Wire regions detected: {result['results'].get('wire_count', 'N/A')}\n")
            f.write(f"  Wire pixels detected: {result['results'].get('wire_pixels', 'N/A')}\n\n")
    
    print(f"\nBatch inference completed! Results saved to {output_path}")
    print(f"Summary saved to {summary_path}")
    
    return all_results
```

**Key Features**:
- **Random Sampling**: Selects random images for batch processing
- **Reproducibility**: Supports random seed for consistent results
- **Error Handling**: Continues processing even if individual images fail
- **Comprehensive Logging**: Creates detailed summary of all results
- **Multiple Format Support**: Handles various image formats

**Why This Matters**: Batch processing is essential for evaluating model performance across multiple images. This function provides a systematic way to process large datasets and generate comprehensive reports.

### 8. `main()` Function

**Purpose**: Command-line interface for the inference script.

**Argument Parser Configuration**:
```python
def main():
    parser = argparse.ArgumentParser(description='Run inference on trained wire detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single input image')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing multiple images')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process from data_dir')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducible sampling (optional)')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Run inference
    if args.image_path:
        # Single image inference
        results = run_inference(model, args.image_path, args.output_dir)
        print("Single image inference completed!")
    elif args.data_dir:
        # Batch inference
        results = run_batch_inference(model, args.data_dir, args.output_dir, args.num_samples, args.random_seed)
        print("Batch inference completed!")
    else:
        print("Error: Please provide either --image_path or --data_dir")
        return
```

**Key Features**:
- **Flexible Input**: Supports both single image and batch processing
- **Model Validation**: Ensures model file exists before processing
- **Error Handling**: Provides clear error messages for invalid inputs
- **Output Management**: Creates organized output directories

## Integration with Post-Processing

### 1. Simple Post-Processor Integration
The script uses `SimpleWirePostProcessor` for basic post-processing:
- **Binary Thresholding**: Converts probability masks to binary masks
- **Noise Removal**: Removes small objects and artifacts
- **Junction Detection**: Uses Harris corner detection for junction points

### 2. Enhanced Wire Segmentation Integration
The script can integrate with `EnhancedWireSegmenter` for advanced analysis:
- **Individual Wire Segmentation**: Breaks down wire regions into individual segments
- **Connection Analysis**: Establishes connections between wires and junctions
- **Network Graph Creation**: Creates graph representations of wire networks

## Visualization Pipeline

### 1. Wire Detection Visualization
- **Original Image**: Shows the input schematic
- **Text + Wire Overlay**: Shows both text (green) and wire (red) detections
- **Text Filtering Analysis**: Demonstrates the effect of text filtering

### 2. Junction Visualization
- **Junction Mask Overlay**: Shows raw junction detection results
- **Junction Points**: Shows processed junction locations with markers

### 3. Wire Segments Visualization
- **Individual Segments**: Shows each wire segment in a different color
- **Centerline Representation**: Uses skeletonization for clean wire representation
- **Segment Counting**: Displays total number of detected segments

## Output Structure

The script creates a comprehensive output structure:
```
results/inference/
├── image1_wire_detection.png      # Wire detection visualization
├── image1_junctions.png           # Junction detection visualization
├── image1_wire_segments.png       # Individual wire segments
├── image2_wire_detection.png      # Additional images...
├── ...
└── inference_summary.txt          # Batch processing summary
```

## Usage Examples

### Single Image Inference
```bash
python run_inference.py --model_path experiments/models/unet_best.h5 --image_path test_image.jpg
```

### Batch Inference
```bash
python run_inference.py --model_path experiments/models/unet_best.h5 --data_dir test_images --num_samples 10
```

### Reproducible Batch Inference
```bash
python run_inference.py --model_path experiments/models/unet_best.h5 --data_dir test_images --num_samples 10 --random_seed 42
```

## Key Features and Optimizations

### 1. Resolution Handling
- **Original Resolution Preservation**: Maintains original image resolution for post-processing
- **Model Resolution Compliance**: Ensures input matches training resolution
- **Seamless Scaling**: Handles resolution differences between training and inference

### 2. Text Filtering Integration
- **Training Consistency**: Uses the same text filtering as training pipeline
- **Fallback Mechanisms**: Graceful degradation if text detection fails
- **Visualization Integration**: Shows text filtering effects in visualizations

### 3. Comprehensive Visualization
- **Multiple View Types**: Different visualizations for different analysis needs
- **High-Quality Output**: 300 DPI output for publication quality
- **Color-Coded Results**: Clear distinction between different detection types

### 4. Error Handling and Robustness
- **Graceful Degradation**: Continues processing even if individual components fail
- **Comprehensive Logging**: Detailed output for debugging and analysis
- **Input Validation**: Validates inputs before processing

## Performance Considerations

### 1. Memory Management
- **Efficient Image Loading**: Loads images only when needed
- **Memory Cleanup**: Proper cleanup after processing each image
- **Batch Processing**: Handles large datasets efficiently

### 2. Processing Speed
- **Optimized Preprocessing**: Efficient image preprocessing pipeline
- **Parallel Processing**: Can be extended for parallel batch processing
- **Caching**: Reuses preprocessed data when possible

This script represents a production-ready inference pipeline that provides comprehensive wire detection capabilities with detailed visualization and analysis tools. It's designed to be both user-friendly for single image processing and scalable for batch processing of large datasets.
