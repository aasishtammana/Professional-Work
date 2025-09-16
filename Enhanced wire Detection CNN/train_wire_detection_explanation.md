# Comprehensive Explanation: train_wire_detection.py

## Overview
The `train_wire_detection.py` script is the central training pipeline for the Enhanced Wire Detection CNN system. This script orchestrates the entire machine learning workflow, from data preparation to model training, evaluation, and result visualization. It's designed to train deep learning models that can detect wires and junctions in circuit schematic images.

## Architecture and Dependencies

### Core Dependencies
- **TensorFlow/Keras**: Deep learning framework for model creation and training
- **OpenCV (cv2)**: Computer vision operations for image processing
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization and plotting
- **scikit-learn**: Data splitting and preprocessing utilities
- **Albumentations**: Advanced image augmentation library
- **psutil**: System resource monitoring
- **GPUtil**: GPU memory monitoring (optional)

### Custom Module Imports
The script imports three critical custom modules:
1. `enhanced_models`: Contains CNN architectures (U-Net, ResNet, Attention-based)
2. `enhanced_data_pipeline`: Handles data loading, preprocessing, and augmentation
3. `custom_annotation_format`: Manages custom annotation format and wire detection

## Detailed Function Analysis

### 1. `setup_gpu()` Function

**Purpose**: Configures GPU settings for optimal training performance and memory management.

**Detailed Logic**:
```python
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). GPU memory growth enabled.")
            
            # Try to set memory limit (only available in newer TF versions)
            try:
                tf.config.experimental.set_memory_limit(gpus[0], 12000)  # 12GB limit
                print("GPU memory limit set to 12GB to prevent OOM.")
            except AttributeError:
                print("GPU memory limit not available in this TensorFlow version.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")
```

**Key Features**:
- **Memory Growth**: Enables dynamic memory allocation to prevent TensorFlow from allocating all GPU memory at once
- **Memory Limiting**: Sets a 12GB limit to prevent out-of-memory (OOM) errors
- **Error Handling**: Gracefully handles cases where GPU configuration fails
- **Fallback**: Automatically falls back to CPU if no GPU is available

**Why This Matters**: GPU memory management is crucial for training large CNN models. Without proper configuration, TensorFlow might allocate all available GPU memory, causing OOM errors or preventing other processes from using the GPU.

### 2. `create_callbacks()` Function

**Purpose**: Creates a comprehensive set of training callbacks for model optimization, monitoring, and memory management.

**Detailed Logic**:

#### Model Checkpoint Callback
```python
checkpoint = ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
```
- **Purpose**: Saves the best model based on validation loss
- **Behavior**: Only saves when validation loss improves
- **Saves**: Complete model (architecture + weights), not just weights

#### TensorBoard Callback
```python
tensorboard = TensorBoard(
    log_dir=str(tensorboard_path),
    histogram_freq=1,
    write_graph=True,
    write_images=True
)
```
- **Purpose**: Enables real-time training monitoring
- **Features**: 
  - Logs training metrics (loss, accuracy)
  - Records weight histograms every epoch
  - Saves model graph structure
  - Captures sample images for visualization

#### Early Stopping Callback
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
```
- **Purpose**: Prevents overfitting by stopping training when validation loss stops improving
- **Patience**: Waits 10 epochs before stopping
- **Restore**: Automatically restores the best weights when stopping

#### Learning Rate Reduction Callback
```python
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
```
- **Purpose**: Dynamically reduces learning rate when training plateaus
- **Factor**: Reduces learning rate by 50% when triggered
- **Patience**: Waits 5 epochs before reducing
- **Minimum**: Prevents learning rate from going below 1e-7

#### Custom Memory Cleanup Callback
```python
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            import gc
            gc.collect()
            tf.keras.backend.clear_session()
            print(f"Memory cleanup performed at epoch {epoch}")
        else:
            import gc
            gc.collect()
```
- **Purpose**: Manages memory usage during long training sessions
- **Heavy Cleanup**: Every 10 epochs, performs full memory cleanup
- **Light Cleanup**: Every epoch, runs garbage collection
- **Session Clearing**: Clears TensorFlow's internal session cache

### 3. `train_model()` Function

**Purpose**: The main training function that orchestrates the entire training pipeline.

**Parameters**:
- `model_type`: Architecture type ('unet', 'resnet', 'attention')
- `data_dir`: Directory containing schematic images
- `annotation_dir`: Directory containing annotations
- `experiment_dir`: Directory to save results
- `batch_size`: Number of samples per batch
- `epochs`: Total number of training epochs
- `image_size`: Input image dimensions
- `learning_rate`: Initial learning rate
- `validation_split`: Fraction of data for validation

**Detailed Workflow**:

#### Step 1: Setup and Directory Creation
```python
setup_gpu()
experiment_dir = Path(experiment_dir)
experiment_dir.mkdir(parents=True, exist_ok=True)
```
- Configures GPU settings
- Creates experiment directory structure

#### Step 2: Annotation Generation
```python
if not Path(annotation_dir).exists():
    print("Generating annotations...")
    create_annotation_batch(
        images_dir=data_dir,
        output_dir=annotation_dir,
        min_wire_length=8,
        border_size=12,
        threshold=0.12
    )
```
- Checks if annotations exist
- If not, generates them using the custom annotation format
- Uses the Annotation Helper module for wire detection

#### Step 3: Data Generator Creation
```python
train_gen, val_gen, dataset = create_data_generators(
    data_dir=data_dir,
    annotation_dir=annotation_dir,
    batch_size=batch_size,
    validation_split=validation_split,
    image_size=image_size
)
```
- Creates training and validation data generators
- Handles data splitting automatically
- Applies preprocessing and augmentation

#### Step 4: Model Creation and Compilation
```python
model = create_model(
    model_type=model_type,
    input_shape=(*image_size, 3)  # RGB images
)
model = compile_model(model, learning_rate=learning_rate)
```
- Creates the specified model architecture
- Compiles with appropriate loss functions and metrics
- Sets up multi-task learning for wires and junctions

#### Step 5: Model Architecture Saving
```python
model_path = experiment_dir / f"{model_type}_architecture.json"
with open(model_path, 'w') as f:
    json.dump(model.to_json(), f, indent=2)
```
- Saves model architecture as JSON
- Enables model reconstruction without code

#### Step 6: Training Execution
```python
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)
```
- Executes the actual training process
- Uses all configured callbacks
- Returns training history for analysis

#### Step 7: Results Saving and Visualization
```python
# Save training history
history_path = experiment_dir / f"{model_type}_history.json"
with open(history_path, 'w') as f:
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    json.dump(history_dict, f, indent=2)

# Plot training history
plot_training_history(history, experiment_dir, model_type)

# Save final model
final_model_path = experiment_dir / "models" / f"{model_type}_final.h5"
model.save(str(final_model_path))
```

### 4. `plot_training_history()` Function

**Purpose**: Creates comprehensive visualization of training progress.

**Detailed Logic**:
```python
def plot_training_history(history, experiment_dir, model_type):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    axes[0, 0].plot(history.history['loss'], label='Total Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Total Loss')
    axes[0, 0].plot(history.history['wire_loss'], label='Wire Loss')
    axes[0, 0].plot(history.history['val_wire_loss'], label='Val Wire Loss')
    axes[0, 0].plot(history.history['junction_loss'], label='Junction Loss')
    axes[0, 0].plot(history.history['val_junction_loss'], label='Val Junction Loss')
    axes[0, 0].set_title('Model Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Wire accuracy
    axes[0, 1].plot(history.history['wire_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_wire_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Wire Mask Accuracy')
    
    # Junction accuracy
    axes[1, 0].plot(history.history['junction_accuracy'], label='Training Accuracy')
    axes[1, 0].plot(history.history['val_junction_accuracy'], label='Validation Accuracy')
    axes[1, 0].set_title('Junction Mask Accuracy')
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
```

**Visualization Components**:
1. **Loss Curves**: Shows total loss, wire loss, and junction loss for both training and validation
2. **Accuracy Curves**: Displays accuracy metrics for wire and junction detection
3. **Learning Rate**: Shows how learning rate changes during training
4. **Multi-task Monitoring**: Tracks both wire and junction detection performance

### 5. `evaluate_model()` Function

**Purpose**: Evaluates a trained model on the validation dataset.

**Detailed Logic**:
```python
def evaluate_model(model_path, data_dir, annotation_dir, batch_size=8, image_size=(512, 512)):
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create data generators
    train_gen, val_gen, dataset = create_data_generators(
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        batch_size=batch_size,
        validation_split=0.2,
        image_size=image_size
    )
    
    # Evaluate on validation set
    results = model.evaluate(val_gen, verbose=1)
    
    # Print results
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")
    
    return results
```

**Key Features**:
- **Model Loading**: Loads pre-trained model from file
- **Data Preparation**: Creates fresh data generators for evaluation
- **Comprehensive Metrics**: Evaluates all configured metrics
- **Detailed Output**: Prints formatted results

### 6. `main()` Function

**Purpose**: Command-line interface and argument parsing.

**Argument Parser Configuration**:
```python
parser = argparse.ArgumentParser(description='Train wire detection model')
parser.add_argument('--model_type', type=str, default='unet', 
                   choices=['unet', 'resnet', 'attention'],
                   help='Type of model to train')
parser.add_argument('--data_dir', type=str, default='data/schematics_dataset',
                   help='Directory containing schematic images')
parser.add_argument('--annotation_dir', type=str, default='annotations',
                   help='Directory containing annotations')
parser.add_argument('--experiment_dir', type=str, default='experiments',
                   help='Directory to save experiment results')
parser.add_argument('--batch_size', type=int, default=4,
                   help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100,
                   help='Number of training epochs')
parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                   help='Input image size (height width)')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                   help='Learning rate for optimizer')
parser.add_argument('--validation_split', type=float, default=0.2,
                   help='Fraction of data for validation')
parser.add_argument('--evaluate', type=str, default=None,
                   help='Path to model to evaluate')
```

**Dual Mode Operation**:
1. **Training Mode**: When `--evaluate` is not provided, trains a new model
2. **Evaluation Mode**: When `--evaluate` is provided, evaluates an existing model

## Memory Management Strategy

The script implements sophisticated memory management to handle large-scale training:

### 1. GPU Memory Management
- **Dynamic Allocation**: Uses memory growth to allocate GPU memory as needed
- **Memory Limiting**: Sets 12GB limit to prevent OOM errors
- **Session Clearing**: Regularly clears TensorFlow session cache

### 2. System Resource Monitoring
```python
def log_memory_usage(epoch):
    ram = psutil.virtual_memory()
    print(f"Epoch {epoch} - RAM: {ram.percent}% used ({ram.used/1024**3:.1f}GB/{ram.total/1024**3:.1f}GB)")
    
    # GPU memory monitoring
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"Epoch {epoch} - GPU: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    except:
        pass
```

### 3. Automatic Cleanup
- **Garbage Collection**: Runs `gc.collect()` every epoch
- **Session Clearing**: Clears TensorFlow session every 3 epochs
- **Progressive Cleanup**: Light cleanup every epoch, heavy cleanup every 10 epochs

## Integration with Custom Modules

### 1. Enhanced Models Integration
The script uses the `enhanced_models` module to create different CNN architectures:
- **U-Net**: Encoder-decoder architecture with skip connections
- **ResNet**: Residual network with identity mappings
- **Attention**: Transformer-based attention mechanisms

### 2. Data Pipeline Integration
The script leverages `enhanced_data_pipeline` for:
- **Data Loading**: Automatic image and annotation loading
- **Preprocessing**: Image normalization and resizing
- **Augmentation**: Real-time data augmentation during training
- **Text Filtering**: OCR-based text region filtering

### 3. Annotation Format Integration
The script uses `custom_annotation_format` for:
- **Wire Detection**: Automated wire detection using Hough transforms
- **Junction Detection**: Intersection point detection
- **Format Conversion**: Converting between different annotation formats

## Training Workflow Summary

1. **Initialization**: Setup GPU, create directories, parse arguments
2. **Data Preparation**: Load images, generate annotations if needed
3. **Model Creation**: Build and compile the specified architecture
4. **Training Setup**: Create callbacks, configure monitoring
5. **Training Execution**: Run the training loop with all optimizations
6. **Results Processing**: Save model, history, and visualizations
7. **Evaluation**: Optionally evaluate the trained model

## Key Features and Optimizations

### 1. Multi-Task Learning
The script trains models to perform two tasks simultaneously:
- **Wire Detection**: Identifying wire segments in schematics
- **Junction Detection**: Finding intersection points between wires

### 2. Advanced Loss Functions
- **Combined Wire Loss**: Binary cross-entropy + Dice loss for thin structures
- **Focal Junction Loss**: Handles class imbalance in junction detection
- **Weighted Losses**: Balances the importance of different tasks

### 3. Comprehensive Monitoring
- **Real-time Metrics**: Training and validation metrics
- **Memory Tracking**: RAM and GPU memory usage
- **Visualization**: Training curves and progress plots
- **TensorBoard**: Detailed training analysis

### 4. Robust Error Handling
- **GPU Fallback**: Automatically falls back to CPU if GPU fails
- **Memory Management**: Prevents OOM errors through careful allocation
- **Annotation Generation**: Handles missing annotations gracefully

## Usage Examples

### Basic Training
```bash
python train_wire_detection.py --model_type unet --epochs 50 --batch_size 8
```

### Advanced Training
```bash
python train_wire_detection.py \
    --model_type attention \
    --data_dir /path/to/images \
    --annotation_dir /path/to/annotations \
    --experiment_dir /path/to/results \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 0.001 \
    --image_size 1024 1024
```

### Model Evaluation
```bash
python train_wire_detection.py \
    --evaluate /path/to/model.h5 \
    --data_dir /path/to/test/images
```

## Output Structure

The script creates a comprehensive output structure:
```
experiments/
├── models/
│   ├── unet_best.h5          # Best model during training
│   └── unet_final.h5         # Final model after training
├── tensorboard/              # TensorBoard logs
├── unet_architecture.json    # Model architecture
├── unet_history.json         # Training history
└── unet_training_history.png # Training curves plot
```

This script represents a production-ready training pipeline that combines modern deep learning techniques with robust engineering practices for training wire detection models on circuit schematics.
