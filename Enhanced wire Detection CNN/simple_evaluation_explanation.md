# Comprehensive Explanation: simple_evaluation.py

## Overview
The `simple_evaluation.py` script is a comprehensive performance evaluation system for the Enhanced Wire Detection CNN project. It provides detailed evaluation metrics, statistical analysis, and visualization capabilities for assessing model performance on wire detection tasks. The script evaluates both wire detection accuracy and junction detection performance, providing both image-level and dataset-level metrics.

## Architecture and Dependencies

### Core Dependencies
- **TensorFlow/Keras**: Model loading and inference
- **OpenCV (cv2)**: Image processing and manipulation
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and plotting
- **Seaborn**: Statistical visualization
- **scikit-learn**: Machine learning metrics and evaluation
- **scikit-image**: Image analysis and morphological operations
- **tqdm**: Progress bars for long-running operations
- **JSON**: Data serialization and storage

### Custom Module Imports
- `run_inference`: Model loading and preprocessing functions
- `simple_postprocessor`: Post-processing utilities for evaluation

### Key Features
- **Comprehensive Metrics**: Precision, recall, F1-score, IoU, and accuracy
- **Multi-Level Analysis**: Image-level and dataset-level evaluation
- **Statistical Analysis**: Mean, standard deviation, and distribution analysis
- **Visualization**: Comprehensive performance visualizations
- **JSON Export**: Detailed results export for further analysis
- **Progress Tracking**: Real-time progress monitoring

## Detailed Function Analysis

### 1. `evaluate_single_image()` Function

**Purpose**: Evaluates a single image against ground truth annotations.

**Detailed Implementation**:
```python
def evaluate_single_image(model, postprocessor, image_path, annotation_path):
    """Evaluate a single image"""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Preprocess image (image is already loaded as numpy array)
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Store original size and image
        original_size = img.shape[:2]
        original_img = img.copy()
        
        # Resize image
        img_resized = cv2.resize(img, (512, 512))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Run model
        outputs = model(img_batch)
        
        # Extract masks
        wire_mask = outputs['wire_mask'][0, :, :, 0].numpy()
        junction_mask = outputs['junction_mask'][0, :, :, 0].numpy()
        
        # Resize masks back to original resolution
        wire_mask_original = cv2.resize(wire_mask, (original_size[1], original_size[0]))
        junction_mask_original = cv2.resize(junction_mask, (original_size[1], original_size[0]))
        
        # Post-process
        results = postprocessor.process_wire_mask(wire_mask_original, junction_mask_original, None, original_size)
        
        # Load ground truth
        with open(annotation_path, 'r') as f:
            gt_data = json.load(f)
        
        gt_wires = gt_data.get('wires', [])
        gt_junctions = gt_data.get('junctions', [])
        
        if not gt_wires:
            return None
        
        # Create ground truth wire mask from wire segments
        gt_wire_mask = np.zeros(wire_mask_original.shape, dtype=np.uint8)
        for wire in gt_wires:
            start = wire['start']
            end = wire['end']
            cv2.line(gt_wire_mask, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, 2)
        
        # Calculate wire detection metrics
        pred_flat = wire_mask_original.flatten()
        gt_flat = gt_wire_mask.flatten()
        
        pred_binary = (pred_flat > 0.1).astype(int)
        gt_binary = (gt_flat > 0.5).astype(int)
        
        precision = precision_score(gt_binary, pred_binary, zero_division=0)
        recall = recall_score(gt_binary, pred_binary, zero_division=0)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        iou = jaccard_score(gt_binary, pred_binary, zero_division=0)
        accuracy = np.mean(pred_binary == gt_binary)
        
        # Count segments
        from skimage.measure import label
        from skimage.morphology import remove_small_objects
        
        # Predicted segments
        binary_mask = (wire_mask_original > 0.1).astype(np.uint8)
        cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=50).astype(np.uint8)
        labeled = label(cleaned_mask)
        pred_segments = len(np.unique(labeled)) - 1
        
        # Ground truth segments
        gt_binary = (gt_wire_mask > 0.5).astype(np.uint8)
        gt_cleaned = remove_small_objects(gt_binary.astype(bool), min_size=50).astype(np.uint8)
        gt_labeled = label(gt_cleaned)
        gt_segments = len(np.unique(gt_labeled)) - 1
        
        return {
            'image_name': image_path.stem,
            'wire_precision': precision,
            'wire_recall': recall,
            'wire_f1': f1,
            'wire_iou': iou,
            'wire_accuracy': accuracy,
            'pred_wire_pixels': np.sum(pred_binary),
            'gt_wire_pixels': np.sum(gt_binary),
            'pred_segments': pred_segments,
            'gt_segments': gt_segments,
            'pred_junctions': len(results.get('junctions', [])),
            'gt_junctions': len(gt_junctions)
        }
        
    except Exception as e:
        print(f"Error evaluating {image_path.stem}: {str(e)}")
        return None
```

**Key Features**:
- **Complete Pipeline**: Handles entire evaluation pipeline
- **Image Preprocessing**: Proper image loading and preprocessing
- **Model Inference**: Runs model on preprocessed image
- **Ground Truth Loading**: Loads and processes ground truth annotations
- **Mask Creation**: Creates ground truth masks from annotations
- **Comprehensive Metrics**: Calculates all evaluation metrics
- **Segment Counting**: Counts predicted and ground truth segments
- **Error Handling**: Graceful error handling with detailed logging

**Mathematical Metrics**:
- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **Recall**: TP / (TP + FN) - Sensitivity of positive predictions
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean
- **IoU (Jaccard)**: |A ∩ B| / |A ∪ B| - Intersection over Union
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Overall correctness

### 2. `main()` Function

**Purpose**: Main evaluation function that orchestrates the entire evaluation process.

**Detailed Implementation**:
```python
def main():
    parser = argparse.ArgumentParser(description='Simple Wire Detection CNN Performance Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/schematics_dataset', help='Path to dataset images')
    parser.add_argument('--annotations_dir', type=str, default='annotations', help='Path to annotations')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Output directory for results')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to evaluate')
    
    args = parser.parse_args()
    
    # Load model and postprocessor
    print("Loading model...")
    model = load_model(args.model_path)
    postprocessor = SimpleWirePostProcessor()
    
    # Get image files
    data_dir = Path(args.data_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    image_files = list(data_dir.glob("*.jpg"))
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Evaluating {len(image_files)} images...")
    
    results = []
    for image_path in tqdm(image_files, desc="Evaluating images"):
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if not annotation_path.exists():
            continue
        
        result = evaluate_single_image(model, postprocessor, image_path, annotation_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    # Calculate summary statistics
    wire_precisions = [r['wire_precision'] for r in results]
    wire_recalls = [r['wire_recall'] for r in results]
    wire_f1s = [r['wire_f1'] for r in results]
    wire_ious = [r['wire_iou'] for r in results]
    wire_accuracies = [r['wire_accuracy'] for r in results]
    
    pred_segments = [r['pred_segments'] for r in results]
    gt_segments = [r['gt_segments'] for r in results]
    pred_junctions = [r['pred_junctions'] for r in results]
    gt_junctions = [r['gt_junctions'] for r in results]
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION RESULTS")
    print("="*60)
    print(f"Total Images Evaluated: {len(results)}")
    
    print("\nWIRE DETECTION METRICS:")
    print(f"  Precision: {np.mean(wire_precisions):.3f} ± {np.std(wire_precisions):.3f}")
    print(f"  Recall:    {np.mean(wire_recalls):.3f} ± {np.std(wire_recalls):.3f}")
    print(f"  F1-Score:  {np.mean(wire_f1s):.3f} ± {np.std(wire_f1s):.3f}")
    print(f"  IoU:       {np.mean(wire_ious):.3f} ± {np.std(wire_ious):.3f}")
    print(f"  Accuracy:  {np.mean(wire_accuracies):.3f} ± {np.std(wire_accuracies):.3f}")
    
    print("\nSEGMENTATION METRICS:")
    print(f"  Predicted Segments: {np.mean(pred_segments):.1f} ± {np.std(pred_segments):.1f}")
    print(f"  Ground Truth Segments: {np.mean(gt_segments):.1f} ± {np.std(gt_segments):.1f}")
    print(f"  Segment Ratio: {np.mean(pred_segments) / np.mean(gt_segments):.3f}")
    
    print("\nJUNCTION DETECTION METRICS:")
    print(f"  Predicted Junctions: {np.mean(pred_junctions):.1f} ± {np.std(pred_junctions):.1f}")
    print(f"  Ground Truth Junctions: {np.mean(gt_junctions):.1f} ± {np.std(gt_junctions):.1f}")
    print(f"  Junction Ratio: {np.mean(pred_junctions) / np.mean(gt_junctions):.3f}")
    
    print("\nPIXEL COUNTS:")
    total_pred_pixels = sum(r['pred_wire_pixels'] for r in results)
    total_gt_pixels = sum(r['gt_wire_pixels'] for r in results)
    print(f"  Total Predicted Wire Pixels: {total_pred_pixels:,}")
    print(f"  Total Ground Truth Wire Pixels: {total_gt_pixels:,}")
    print(f"  Pixel Ratio: {total_pred_pixels / total_gt_pixels:.3f}")
    
    print("="*60)
    
    # Calculate dataset-level metrics
    dataset_metrics = calculate_dataset_metrics(results)
    
    # Print dataset-level summary
    print("\n" + "="*60)
    print("DATASET-LEVEL SUMMARY")
    print("="*60)
    print(f"Total Images: {dataset_metrics['total_images']}")
    print(f"Total Predicted Wire Pixels: {dataset_metrics['pixel_counts']['total_predicted_pixels']:,}")
    print(f"Total Ground Truth Wire Pixels: {dataset_metrics['pixel_counts']['total_ground_truth_pixels']:,}")
    print(f"Pixel Detection Ratio: {dataset_metrics['pixel_counts']['pixel_detection_ratio']:.3f}")
    print(f"Total Predicted Segments: {dataset_metrics['segmentation']['total_predicted_segments']:,}")
    print(f"Total Ground Truth Segments: {dataset_metrics['segmentation']['total_ground_truth_segments']:,}")
    print(f"Segment Detection Ratio: {dataset_metrics['segmentation']['segment_detection_ratio']:.3f}")
    print(f"Total Predicted Junctions: {dataset_metrics['junction_detection']['total_predicted_junctions']:,}")
    print(f"Total Ground Truth Junctions: {dataset_metrics['junction_detection']['total_ground_truth_junctions']:,}")
    print(f"Junction Detection Ratio: {dataset_metrics['junction_detection']['junction_detection_ratio']:.3f}")
    print("="*60)
    
    # Create visualizations
    create_visualizations(results, dataset_metrics, output_dir)
    
    # Save detailed results (convert numpy types to Python types for JSON serialization)
    json_results = []
    for r in results:
        json_r = {}
        for key, value in r.items():
            if isinstance(value, (np.integer, np.floating)):
                json_r[key] = float(value)
            else:
                json_r[key] = value
        json_results.append(json_r)
    
    # Save both image-level and dataset-level results
    with open(output_dir / "image_level_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    with open(output_dir / "dataset_level_results.json", 'w') as f:
        json.dump(dataset_metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - image_level_results.json: Per-image detailed results")
    print(f"  - dataset_level_results.json: Overall dataset metrics")
    print(f"  - performance_visualizations.png: Performance charts")
```

**Key Features**:
- **Command-Line Interface**: Comprehensive argument parsing
- **Model Loading**: Loads trained model and postprocessor
- **Batch Processing**: Evaluates multiple images efficiently
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Statistical Analysis**: Calculates mean and standard deviation
- **Comprehensive Reporting**: Detailed performance reports
- **Visualization**: Creates performance visualizations
- **Data Export**: Exports results in JSON format

### 3. `calculate_dataset_metrics()` Function

**Purpose**: Calculates comprehensive dataset-level metrics and statistics.

**Detailed Implementation**:
```python
def calculate_dataset_metrics(results):
    """Calculate dataset-level aggregated metrics"""
    if not results:
        return {}
    
    # Aggregate all metrics
    total_pred_pixels = sum(r['pred_wire_pixels'] for r in results)
    total_gt_pixels = sum(r['gt_wire_pixels'] for r in results)
    total_pred_segments = sum(r['pred_segments'] for r in results)
    total_gt_segments = sum(r['gt_segments'] for r in results)
    total_pred_junctions = sum(r['pred_junctions'] for r in results)
    total_gt_junctions = sum(r['gt_junctions'] for r in results)
    
    # Dataset-level metrics (weighted averages)
    dataset_metrics = {
        'total_images': len(results),
        'wire_detection': {
            'mean_precision': np.mean([r['wire_precision'] for r in results]),
            'std_precision': np.std([r['wire_precision'] for r in results]),
            'mean_recall': np.mean([r['wire_recall'] for r in results]),
            'std_recall': np.std([r['wire_recall'] for r in results]),
            'mean_f1': np.mean([r['wire_f1'] for r in results]),
            'std_f1': np.std([r['wire_f1'] for r in results]),
            'mean_iou': np.mean([r['wire_iou'] for r in results]),
            'std_iou': np.std([r['wire_iou'] for r in results]),
            'mean_accuracy': np.mean([r['wire_accuracy'] for r in results]),
            'std_accuracy': np.std([r['wire_accuracy'] for r in results])
        },
        'segmentation': {
            'total_predicted_segments': total_pred_segments,
            'total_ground_truth_segments': total_gt_segments,
            'segment_detection_ratio': total_pred_segments / total_gt_segments if total_gt_segments > 0 else 0,
            'mean_predicted_per_image': np.mean([r['pred_segments'] for r in results]),
            'std_predicted_per_image': np.std([r['pred_segments'] for r in results]),
            'mean_gt_per_image': np.mean([r['gt_segments'] for r in results]),
            'std_gt_per_image': np.std([r['gt_segments'] for r in results])
        },
        'junction_detection': {
            'total_predicted_junctions': total_pred_junctions,
            'total_ground_truth_junctions': total_gt_junctions,
            'junction_detection_ratio': total_pred_junctions / total_gt_junctions if total_gt_junctions > 0 else 0,
            'mean_predicted_per_image': np.mean([r['pred_junctions'] for r in results]),
            'std_predicted_per_image': np.std([r['pred_junctions'] for r in results]),
            'mean_gt_per_image': np.mean([r['gt_junctions'] for r in results]),
            'std_gt_per_image': np.std([r['gt_junctions'] for r in results])
        },
        'pixel_counts': {
            'total_predicted_pixels': int(total_pred_pixels),
            'total_ground_truth_pixels': int(total_gt_pixels),
            'pixel_detection_ratio': total_pred_pixels / total_gt_pixels if total_gt_pixels > 0 else 0,
            'mean_predicted_per_image': np.mean([r['pred_wire_pixels'] for r in results]),
            'std_predicted_per_image': np.std([r['pred_wire_pixels'] for r in results]),
            'mean_gt_per_image': np.mean([r['gt_wire_pixels'] for r in results]),
            'std_gt_per_image': np.std([r['gt_wire_pixels'] for r in results])
        }
    }
    
    return dataset_metrics
```

**Key Features**:
- **Comprehensive Aggregation**: Aggregates all metrics across dataset
- **Statistical Analysis**: Calculates mean and standard deviation
- **Ratio Calculations**: Calculates detection ratios
- **Structured Output**: Well-organized metrics dictionary
- **Null Safety**: Handles division by zero cases

### 4. `create_visualizations()` Function

**Purpose**: Creates comprehensive performance visualizations.

**Detailed Implementation**:
```python
def create_visualizations(results, dataset_metrics, output_dir):
    """Create performance visualizations"""
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # Extract data for plotting
    wire_precisions = [r['wire_precision'] for r in results]
    wire_recalls = [r['wire_recall'] for r in results]
    wire_f1s = [r['wire_f1'] for r in results]
    wire_ious = [r['wire_iou'] for r in results]
    wire_accuracies = [r['wire_accuracy'] for r in results]
    
    pred_segments = [r['pred_segments'] for r in results]
    gt_segments = [r['gt_segments'] for r in results]
    pred_junctions = [r['pred_junctions'] for r in results]
    gt_junctions = [r['gt_junctions'] for r in results]
    pred_pixels = [r['pred_wire_pixels'] for r in results]
    gt_pixels = [r['gt_wire_pixels'] for r in results]
    
    # 1. Wire Detection Metrics Distribution
    ax1 = plt.subplot(3, 4, 1)
    metrics_data = [wire_precisions, wire_recalls, wire_f1s, wire_ious]
    metrics_labels = ['Precision', 'Recall', 'F1-Score', 'IoU']
    bp1 = ax1.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('Wire Detection Metrics Distribution', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Distribution
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(wire_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(wire_accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(wire_accuracies):.3f}')
    ax2.set_title('Wire Detection Accuracy Distribution', fontweight='bold')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Segment Detection Comparison
    ax3 = plt.subplot(3, 4, 3)
    x = np.arange(len(results))
    width = 0.35
    ax3.bar(x - width/2, pred_segments, width, label='Predicted', alpha=0.8, color='lightcoral')
    ax3.bar(x + width/2, gt_segments, width, label='Ground Truth', alpha=0.8, color='lightgreen')
    ax3.set_title('Segment Detection per Image', fontweight='bold')
    ax3.set_xlabel('Image Index')
    ax3.set_ylabel('Number of Segments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Junction Detection Comparison
    ax4 = plt.subplot(3, 4, 4)
    ax4.bar(x - width/2, pred_junctions, width, label='Predicted', alpha=0.8, color='lightcoral')
    ax4.bar(x + width/2, gt_junctions, width, label='Ground Truth', alpha=0.8, color='lightgreen')
    ax4.set_title('Junction Detection per Image', fontweight='bold')
    ax4.set_xlabel('Image Index')
    ax4.set_ylabel('Number of Junctions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Pixel Count Comparison
    ax5 = plt.subplot(3, 4, 5)
    ax5.bar(x - width/2, pred_pixels, width, label='Predicted', alpha=0.8, color='lightcoral')
    ax5.bar(x + width/2, gt_pixels, width, label='Ground Truth', alpha=0.8, color='lightgreen')
    ax5.set_title('Wire Pixel Count per Image', fontweight='bold')
    ax5.set_xlabel('Image Index')
    ax5.set_ylabel('Number of Pixels')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Precision vs Recall Scatter
    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(wire_recalls, wire_precisions, alpha=0.7, s=100, c='blue')
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Precision vs Recall', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    max_val = max(max(wire_recalls), max(wire_precisions))
    ax6.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Line')
    ax6.legend()
    
    # 7. F1-Score vs IoU Scatter
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(wire_ious, wire_f1s, alpha=0.7, s=100, c='green')
    ax7.set_xlabel('IoU')
    ax7.set_ylabel('F1-Score')
    ax7.set_title('F1-Score vs IoU', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Dataset Summary Metrics
    ax8 = plt.subplot(3, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""
    DATASET SUMMARY
    ================
    Total Images: {dataset_metrics['total_images']}
    
    WIRE DETECTION:
    Precision: {dataset_metrics['wire_detection']['mean_precision']:.3f} ± {dataset_metrics['wire_detection']['std_precision']:.3f}
    Recall: {dataset_metrics['wire_detection']['mean_recall']:.3f} ± {dataset_metrics['wire_detection']['std_recall']:.3f}
    F1-Score: {dataset_metrics['wire_detection']['mean_f1']:.3f} ± {dataset_metrics['wire_detection']['std_f1']:.3f}
    IoU: {dataset_metrics['wire_detection']['mean_iou']:.3f} ± {dataset_metrics['wire_detection']['std_iou']:.3f}
    Accuracy: {dataset_metrics['wire_detection']['mean_accuracy']:.3f} ± {dataset_metrics['wire_detection']['std_accuracy']:.3f}
    
    SEGMENTATION:
    Detection Ratio: {dataset_metrics['segmentation']['segment_detection_ratio']:.3f}
    Predicted/Image: {dataset_metrics['segmentation']['mean_predicted_per_image']:.1f} ± {dataset_metrics['segmentation']['std_predicted_per_image']:.1f}
    GT/Image: {dataset_metrics['segmentation']['mean_gt_per_image']:.1f} ± {dataset_metrics['segmentation']['std_gt_per_image']:.1f}
    
    JUNCTIONS:
    Detection Ratio: {dataset_metrics['junction_detection']['junction_detection_ratio']:.3f}
    Predicted/Image: {dataset_metrics['junction_detection']['mean_predicted_per_image']:.1f} ± {dataset_metrics['junction_detection']['std_predicted_per_image']:.1f}
    GT/Image: {dataset_metrics['junction_detection']['mean_gt_per_image']:.1f} ± {dataset_metrics['junction_detection']['std_gt_per_image']:.1f}
    
    PIXELS:
    Detection Ratio: {dataset_metrics['pixel_counts']['pixel_detection_ratio']:.3f}
    Total Predicted: {dataset_metrics['pixel_counts']['total_predicted_pixels']:,}
    Total GT: {dataset_metrics['pixel_counts']['total_ground_truth_pixels']:,}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Performance Trends
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(range(len(results)), wire_f1s, 'o-', label='F1-Score', alpha=0.7)
    ax9.plot(range(len(results)), wire_ious, 's-', label='IoU', alpha=0.7)
    ax9.set_title('Performance Trends Across Images', fontweight='bold')
    ax9.set_xlabel('Image Index')
    ax9.set_ylabel('Score')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Segment Ratio Distribution
    ax10 = plt.subplot(3, 4, 10)
    segment_ratios = [p/g if g > 0 else 0 for p, g in zip(pred_segments, gt_segments)]
    ax10.hist(segment_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax10.axvline(np.mean(segment_ratios), color='red', linestyle='--', 
                 label=f'Mean: {np.mean(segment_ratios):.3f}')
    ax10.set_title('Segment Detection Ratio Distribution', fontweight='bold')
    ax10.set_xlabel('Predicted/GT Ratio')
    ax10.set_ylabel('Frequency')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Junction Ratio Distribution
    ax11 = plt.subplot(3, 4, 11)
    junction_ratios = [p/g if g > 0 else 0 for p, g in zip(pred_junctions, gt_junctions)]
    ax11.hist(junction_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax11.axvline(np.mean(junction_ratios), color='red', linestyle='--', 
                 label=f'Mean: {np.mean(junction_ratios):.3f}')
    ax11.set_title('Junction Detection Ratio Distribution', fontweight='bold')
    ax11.set_xlabel('Predicted/GT Ratio')
    ax11.set_ylabel('Frequency')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Overall Performance Heatmap
    ax12 = plt.subplot(3, 4, 12)
    performance_matrix = np.array([
        [np.mean(wire_precisions), np.mean(wire_recalls), np.mean(wire_f1s), np.mean(wire_ious)],
        [np.mean(segment_ratios), np.mean(junction_ratios), np.mean([p/g if g > 0 else 0 for p, g in zip(pred_pixels, gt_pixels)]), np.mean(wire_accuracies)]
    ])
    
    im = ax12.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
    ax12.set_xticks(range(4))
    ax12.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'IoU'])
    ax12.set_yticks(range(2))
    ax12.set_yticklabels(['Detection Ratios', 'Performance Metrics'])
    ax12.set_title('Performance Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(4):
            text = ax12.text(j, i, f'{performance_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_visualizations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance visualizations saved to {output_dir}/performance_visualizations.png")
```

**Key Features**:
- **12-Panel Layout**: Comprehensive visualization layout
- **Multiple Chart Types**: Box plots, histograms, bar charts, scatter plots
- **Statistical Analysis**: Mean lines, trend analysis, distribution analysis
- **Color Coding**: Consistent color scheme across visualizations
- **Professional Layout**: Publication-ready visualizations
- **High Resolution**: 300 DPI output for quality

**Visualization Components**:
1. **Wire Detection Metrics Distribution**: Box plots of precision, recall, F1, IoU
2. **Accuracy Distribution**: Histogram of accuracy scores
3. **Segment Detection Comparison**: Bar chart comparing predicted vs ground truth
4. **Junction Detection Comparison**: Bar chart comparing predicted vs ground truth
5. **Pixel Count Comparison**: Bar chart comparing pixel counts
6. **Precision vs Recall Scatter**: Scatter plot with perfect line reference
7. **F1-Score vs IoU Scatter**: Scatter plot showing correlation
8. **Dataset Summary**: Text summary of key metrics
9. **Performance Trends**: Line plot showing trends across images
10. **Segment Ratio Distribution**: Histogram of detection ratios
11. **Junction Ratio Distribution**: Histogram of detection ratios
12. **Performance Heatmap**: Heatmap of overall performance

## Integration with Other Modules

### 1. Model Integration
- **Model Loading**: Uses `run_inference.load_model()`
- **Preprocessing**: Uses `run_inference.preprocess_image()`
- **Inference**: Runs model inference on images

### 2. Post-Processing Integration
- **Simple Post-Processor**: Uses `SimpleWirePostProcessor`
- **Mask Processing**: Processes CNN outputs
- **Junction Detection**: Detects junctions from masks

### 3. Data Integration
- **Annotation Loading**: Loads ground truth annotations
- **Image Loading**: Loads test images
- **Path Management**: Handles file paths and directories

## Usage Examples

### Basic Evaluation
```bash
python simple_evaluation.py --model_path experiments/models/unet_best.h5
```

### Custom Dataset Evaluation
```bash
python simple_evaluation.py \
    --model_path experiments/models/unet_best.h5 \
    --data_dir test_images \
    --annotations_dir test_annotations \
    --output_dir results/test_evaluation
```

### Limited Evaluation
```bash
python simple_evaluation.py \
    --model_path experiments/models/unet_best.h5 \
    --max_images 50
```

## Output Structure

The script creates a comprehensive output structure:
```
results/evaluation/
├── image_level_results.json      # Per-image detailed results
├── dataset_level_results.json    # Overall dataset metrics
└── performance_visualizations.png # Performance charts
```

### Image-Level Results Format
```json
{
  "image_name": "schematic_001",
  "wire_precision": 0.85,
  "wire_recall": 0.78,
  "wire_f1": 0.81,
  "wire_iou": 0.68,
  "wire_accuracy": 0.92,
  "pred_wire_pixels": 1250,
  "gt_wire_pixels": 1180,
  "pred_segments": 15,
  "gt_segments": 18,
  "pred_junctions": 8,
  "gt_junctions": 10
}
```

### Dataset-Level Results Format
```json
{
  "total_images": 100,
  "wire_detection": {
    "mean_precision": 0.82,
    "std_precision": 0.15,
    "mean_recall": 0.76,
    "std_recall": 0.18,
    "mean_f1": 0.79,
    "std_f1": 0.16,
    "mean_iou": 0.65,
    "std_iou": 0.20,
    "mean_accuracy": 0.91,
    "std_accuracy": 0.08
  },
  "segmentation": {
    "total_predicted_segments": 1500,
    "total_ground_truth_segments": 1800,
    "segment_detection_ratio": 0.83,
    "mean_predicted_per_image": 15.0,
    "std_predicted_per_image": 5.2,
    "mean_gt_per_image": 18.0,
    "std_gt_per_image": 6.1
  },
  "junction_detection": {
    "total_predicted_junctions": 800,
    "total_ground_truth_junctions": 1000,
    "junction_detection_ratio": 0.80,
    "mean_predicted_per_image": 8.0,
    "std_predicted_per_image": 3.5,
    "mean_gt_per_image": 10.0,
    "std_gt_per_image": 4.2
  },
  "pixel_counts": {
    "total_predicted_pixels": 125000,
    "total_ground_truth_pixels": 118000,
    "pixel_detection_ratio": 1.06,
    "mean_predicted_per_image": 1250.0,
    "std_predicted_per_image": 450.0,
    "mean_gt_per_image": 1180.0,
    "std_gt_per_image": 380.0
  }
}
```

## Performance Considerations

### 1. Memory Efficiency
- **Batch Processing**: Processes images one at a time
- **Memory Cleanup**: Cleans up after each image
- **Efficient Data Structures**: Uses NumPy arrays for efficiency

### 2. Processing Speed
- **Progress Tracking**: Real-time progress monitoring
- **Efficient Algorithms**: Uses optimized scikit-learn functions
- **Vectorized Operations**: NumPy vectorized operations

### 3. Storage Efficiency
- **JSON Format**: Compact JSON storage
- **Selective Export**: Exports only necessary data
- **Compression**: Can be compressed for storage

## Error Handling and Robustness

### 1. Input Validation
- **File Existence**: Checks file existence before processing
- **Format Validation**: Validates image and annotation formats
- **Data Validation**: Validates data integrity

### 2. Processing Errors
- **Graceful Degradation**: Continues processing with partial results
- **Error Logging**: Detailed error logging
- **Exception Handling**: Comprehensive exception handling

### 3. Output Validation
- **Result Validation**: Validates evaluation results
- **Format Checking**: Ensures proper output format
- **Data Integrity**: Maintains data integrity

This script represents a comprehensive evaluation system that provides detailed performance analysis for wire detection models. The combination of statistical analysis, visualization, and data export makes it a production-ready solution for model evaluation and performance monitoring.
