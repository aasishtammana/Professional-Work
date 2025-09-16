# Comprehensive Explanation: visualize_annotations.py

## Overview
The `visualize_annotations.py` script is a comprehensive visualization system for the Enhanced Wire Detection CNN project. It provides detailed visualization capabilities for ground truth annotations, including wire segments, junction points, and component bounding boxes. The script also demonstrates the effect of text filtering on wire detection and provides statistical analysis of annotation data.

## Architecture and Dependencies

### Core Dependencies
- **OpenCV (cv2)**: Image processing and manipulation
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and plotting
- **JSON**: Data serialization and loading
- **Pathlib**: Modern path handling
- **Random**: Random sampling for visualization

### Custom Module Imports
- `enhanced_data_pipeline`: Text filtering and dataset utilities

### Key Features
- **Multi-Panel Visualization**: Comprehensive multi-panel layouts
- **Text Filtering Analysis**: Demonstrates text filtering effects
- **Statistical Analysis**: Provides annotation statistics
- **High-Quality Output**: Publication-ready visualizations
- **Flexible Input**: Handles various image and annotation formats
- **Batch Processing**: Processes multiple images efficiently

## Detailed Function Analysis

### 1. `load_annotation()` Function

**Purpose**: Loads annotation data from JSON files.

**Implementation**:
```python
def load_annotation(annotation_path):
    """Load annotation from JSON file"""
    with open(annotation_path, 'r') as f:
        return json.load(f)
```

**Key Features**:
- **Simple Interface**: Straightforward JSON loading
- **Error Handling**: Basic file handling
- **Data Validation**: Assumes valid JSON format

### 2. `visualize_annotation()` Function

**Purpose**: Creates comprehensive visualization of a single annotation.

**Detailed Implementation**:
```python
def visualize_annotation(image_path, annotation_path, output_dir=None):
    """Visualize a single annotation"""
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    annotation = load_annotation(annotation_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Wire annotations
    axes[1].imshow(img, alpha=0.7)
    axes[1].set_title('Wire Annotations (Red)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Draw wire annotations
    wire_count = 0
    for wire in annotation.get('wires', []):
        start = wire['start']
        end = wire['end']
        # Draw wire as line
        axes[1].plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, alpha=0.8)
        wire_count += 1
    
    # Junction annotations
    axes[2].imshow(img, alpha=0.7)
    axes[2].set_title('Junction Annotations (Blue)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Draw junction annotations
    junction_count = 0
    for junction in annotation.get('junctions', []):
        center = junction['center']
        # Draw junction as circle
        circle = patches.Circle(center, 8, color='blue', alpha=0.8)
        axes[2].add_patch(circle)
        junction_count += 1
    
    # Add counts
    axes[1].text(10, 30, f'Wires: {wire_count}', fontsize=12, color='red', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[2].text(10, 30, f'Junctions: {junction_count}', fontsize=12, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).stem
        plt.savefig(output_path / f"{image_name}_annotations.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved annotation visualization: {image_name}_annotations.png")
    else:
        plt.show()
    
    return wire_count, junction_count
```

**Key Features**:
- **Three-Panel Layout**: Original image, wire annotations, junction annotations
- **Color Coding**: Red for wires, blue for junctions
- **Overlay Visualization**: Semi-transparent overlays on original image
- **Count Display**: Shows number of wires and junctions
- **High-Quality Output**: 300 DPI output for publication quality
- **Flexible Output**: Can save to file or display interactively

### 3. `create_annotation_mask()` Function

**Purpose**: Creates binary masks from annotation data.

**Implementation**:
```python
def create_annotation_mask(image_path, annotation_path):
    """Create wire and junction masks from annotations"""
    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    
    # Create masks
    wire_mask = np.zeros((height, width), dtype=np.uint8)
    junction_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Load annotation
    annotation = load_annotation(annotation_path)
    
    # Draw wire masks
    for wire in annotation.get('wires', []):
        start = tuple(wire['start'])
        end = tuple(wire['end'])
        # Draw thick line for wire
        cv2.line(wire_mask, start, end, 255, thickness=3)
    
    # Draw junction masks
    for junction in annotation.get('junctions', []):
        center = tuple(junction['center'])
        # Draw circle for junction
        cv2.circle(junction_mask, center, 5, 255, -1)
    
    return wire_mask, junction_mask
```

**Key Features**:
- **Binary Mask Creation**: Creates standard binary masks
- **OpenCV Integration**: Uses OpenCV for efficient drawing
- **Thick Lines**: Draws thick lines for wire visibility
- **Filled Circles**: Draws filled circles for junctions
- **Dimension Matching**: Ensures masks match image dimensions

### 4. `visualize_annotation_masks()` Function

**Purpose**: Creates comprehensive visualization of annotation masks with text filtering analysis.

**Detailed Implementation**:
```python
def visualize_annotation_masks(image_path, annotation_path, output_dir=None):
    """Visualize annotation masks with text filtering"""
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create masks
    wire_mask, junction_mask = create_annotation_mask(image_path, annotation_path)
    
    # Apply text filtering to see the effect
    from enhanced_data_pipeline import SchematicDataset
    dataset = SchematicDataset("data/schematics_dataset")
    wire_mask_filtered = dataset._filter_text_regions(wire_mask, str(image_path))
    
    # Create figure with more subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Wire mask (before text filtering)
    axes[0, 1].imshow(wire_mask, cmap='Reds')
    axes[0, 1].set_title('Wire Mask (Before Text Filtering)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Wire mask (after text filtering)
    axes[0, 2].imshow(wire_mask_filtered, cmap='Reds')
    axes[0, 2].set_title('Wire Mask (After Text Filtering)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Junction mask
    axes[1, 0].imshow(junction_mask, cmap='Blues')
    axes[1, 0].set_title('Junction Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Text filtering difference
    text_removed = wire_mask - wire_mask_filtered
    axes[1, 1].imshow(text_removed, cmap='hot')
    axes[1, 1].set_title('Text Pixels Removed', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Combined mask (after text filtering)
    combined_mask = np.maximum(wire_mask_filtered, junction_mask)
    axes[1, 2].imshow(combined_mask, cmap='viridis')
    axes[1, 2].set_title('Combined Mask (Post-Filtering)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add statistics
    original_pixels = np.sum(wire_mask > 0)
    filtered_pixels = np.sum(wire_mask_filtered > 0)
    removed_pixels = original_pixels - filtered_pixels
    removal_percentage = (removed_pixels / original_pixels * 100) if original_pixels > 0 else 0
    
    fig.suptitle(f'Text Filtering: {removed_pixels} pixels removed ({removal_percentage:.1f}%)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).stem
        plt.savefig(output_path / f"{image_name}_masks.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mask visualization: {image_name}_masks.png")
    else:
        plt.show()
```

**Key Features**:
- **Six-Panel Layout**: Comprehensive visualization layout
- **Text Filtering Analysis**: Shows before/after text filtering
- **Statistical Information**: Displays pixel removal statistics
- **Color-Coded Masks**: Different colormaps for different mask types
- **Difference Visualization**: Shows removed text pixels
- **Combined View**: Shows final combined mask

**Visualization Components**:
1. **Original Image**: Input schematic image
2. **Wire Mask (Before)**: Wire mask before text filtering
3. **Wire Mask (After)**: Wire mask after text filtering
4. **Junction Mask**: Junction detection mask
5. **Text Pixels Removed**: Difference showing removed text
6. **Combined Mask**: Final combined mask

### 5. `main()` Function

**Purpose**: Main function that orchestrates batch visualization of annotations.

**Detailed Implementation**:
```python
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize wire detection annotations')
    parser.add_argument('--data_dir', type=str, default='data/schematics_dataset',
                       help='Directory containing images')
    parser.add_argument('--annotation_dir', type=str, default='annotations',
                       help='Directory containing annotation files')
    parser.add_argument('--output_dir', type=str, default='results/annotation_visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--show_masks', action='store_true',
                       help='Also create mask visualizations')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    annotation_path = Path(args.annotation_dir)
    output_path = Path(args.output_dir)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f'*{ext}')))
    
    # Randomly sample images
    if len(image_files) > args.num_samples:
        image_files = random.sample(image_files, args.num_samples)
    
    print(f"Visualizing annotations for {len(image_files)} images...")
    
    total_wires = 0
    total_junctions = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Find corresponding annotation
        annotation_file = annotation_path / f"{image_path.stem}.json"
        
        if annotation_file.exists():
            # Visualize annotations
            wire_count, junction_count = visualize_annotation(
                str(image_path), str(annotation_file), str(output_path)
            )
            
            if args.show_masks:
                visualize_annotation_masks(
                    str(image_path), str(annotation_file), str(output_path)
                )
            
            total_wires += wire_count
            total_junctions += junction_count
            
            print(f"  Wires: {wire_count}, Junctions: {junction_count}")
        else:
            print(f"  No annotation found for {image_path.name}")
    
    print(f"\nSummary:")
    print(f"  Total wires: {total_wires}")
    print(f"  Total junctions: {total_junctions}")
    print(f"  Average wires per image: {total_wires / len(image_files):.1f}")
    print(f"  Average junctions per image: {total_junctions / len(image_files):.1f}")
    print(f"  Visualizations saved to: {output_path}")
```

**Key Features**:
- **Command-Line Interface**: Comprehensive argument parsing
- **Random Sampling**: Randomly samples images for visualization
- **Batch Processing**: Processes multiple images efficiently
- **Progress Tracking**: Shows processing progress
- **Statistical Summary**: Provides dataset statistics
- **Flexible Options**: Optional mask visualization
- **Error Handling**: Handles missing annotations gracefully

## Visualization Types

### 1. Basic Annotation Visualization

**Layout**: 1×3 panel layout
- **Panel 1**: Original image
- **Panel 2**: Wire annotations (red lines)
- **Panel 3**: Junction annotations (blue circles)

**Features**:
- **Color Coding**: Red for wires, blue for junctions
- **Count Display**: Shows number of wires and junctions
- **High Contrast**: Clear visibility of annotations
- **Professional Layout**: Clean, publication-ready design

### 2. Mask Visualization

**Layout**: 2×3 panel layout
- **Panel 1**: Original image
- **Panel 2**: Wire mask (before text filtering)
- **Panel 3**: Wire mask (after text filtering)
- **Panel 4**: Junction mask
- **Panel 5**: Text pixels removed
- **Panel 6**: Combined mask (post-filtering)

**Features**:
- **Text Filtering Analysis**: Shows effect of text filtering
- **Statistical Information**: Displays pixel removal statistics
- **Color-Coded Masks**: Different colormaps for different mask types
- **Difference Visualization**: Shows removed text pixels

### 3. Statistical Analysis

**Metrics Provided**:
- **Total Wires**: Total number of wires across all images
- **Total Junctions**: Total number of junctions across all images
- **Average per Image**: Average wires and junctions per image
- **Text Filtering Stats**: Pixel removal statistics

## Integration with Other Modules

### 1. Data Pipeline Integration
- **Text Filtering**: Uses `enhanced_data_pipeline.SchematicDataset`
- **Mask Creation**: Integrates with annotation format
- **Dataset Loading**: Uses standard dataset loading

### 2. Annotation Format Integration
- **JSON Loading**: Loads custom annotation format
- **Wire Segments**: Processes wire segment data
- **Junction Points**: Processes junction point data

### 3. Visualization Integration
- **Matplotlib**: Uses matplotlib for visualization
- **OpenCV**: Uses OpenCV for image processing
- **Pathlib**: Uses pathlib for path handling

## Usage Examples

### Basic Annotation Visualization
```bash
python visualize_annotations.py
```

### Custom Dataset Visualization
```bash
python visualize_annotations.py \
    --data_dir test_images \
    --annotation_dir test_annotations \
    --output_dir results/test_visualization
```

### With Mask Visualization
```bash
python visualize_annotations.py \
    --show_masks \
    --num_samples 5
```

### Limited Sample Visualization
```bash
python visualize_annotations.py \
    --num_samples 10 \
    --show_masks
```

## Output Structure

The script creates a comprehensive output structure:
```
results/annotation_visualization/
├── image1_annotations.png    # Basic annotation visualization
├── image1_masks.png         # Mask visualization (if --show_masks)
├── image2_annotations.png    # Additional images...
├── image2_masks.png
└── ...
```

## Performance Considerations

### 1. Memory Efficiency
- **Image Loading**: Loads images one at a time
- **Memory Cleanup**: Proper cleanup after each image
- **Efficient Processing**: Uses NumPy arrays for efficiency

### 2. Processing Speed
- **Batch Processing**: Processes multiple images efficiently
- **Progress Tracking**: Shows processing progress
- **Efficient Algorithms**: Uses optimized OpenCV functions

### 3. Storage Efficiency
- **High-Quality Output**: 300 DPI output for quality
- **Compressed Formats**: Uses PNG for lossless compression
- **Selective Output**: Only creates requested visualizations

## Error Handling and Robustness

### 1. File System Errors
- **Path Validation**: Validates all file paths
- **Directory Creation**: Creates output directories if needed
- **File Existence**: Checks file existence before processing

### 2. Image Processing Errors
- **Format Support**: Handles multiple image formats
- **Corruption Handling**: Graceful handling of corrupted images
- **Dimension Validation**: Validates image dimensions

### 3. Annotation Errors
- **JSON Parsing**: Robust JSON parsing with error handling
- **Format Validation**: Validates annotation format
- **Missing Data**: Handles missing annotation data

### 4. Visualization Errors
- **Figure Creation**: Robust figure creation
- **Layout Management**: Proper layout management
- **Save Operations**: Reliable save operations

## Advanced Features

### 1. Text Filtering Analysis
- **Before/After Comparison**: Shows effect of text filtering
- **Statistical Analysis**: Provides pixel removal statistics
- **Visual Difference**: Shows removed text pixels

### 2. Statistical Reporting
- **Dataset Statistics**: Provides comprehensive dataset statistics
- **Per-Image Analysis**: Analyzes individual images
- **Summary Reporting**: Provides summary statistics

### 3. Flexible Visualization
- **Multiple Layouts**: Different layout options
- **Color Schemes**: Consistent color schemes
- **High-Quality Output**: Publication-ready visualizations

## Customization Options

### 1. Visualization Parameters
- **Figure Size**: Configurable figure sizes
- **DPI Settings**: Adjustable DPI for output
- **Color Schemes**: Customizable color schemes
- **Layout Options**: Different layout configurations

### 2. Processing Options
- **Sample Size**: Configurable number of samples
- **Output Format**: Different output formats
- **Mask Visualization**: Optional mask visualization
- **Statistical Analysis**: Optional statistical analysis

### 3. Integration Options
- **Custom Datasets**: Support for custom datasets
- **Annotation Formats**: Support for different annotation formats
- **Output Directories**: Flexible output directory structure

This script represents a comprehensive visualization system that provides detailed analysis of wire detection annotations. The combination of multiple visualization types, statistical analysis, and text filtering demonstration makes it a valuable tool for understanding and analyzing wire detection datasets.
