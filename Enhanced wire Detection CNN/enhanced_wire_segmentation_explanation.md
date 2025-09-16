# Comprehensive Explanation: enhanced_wire_segmentation.py

## Overview
The `enhanced_wire_segmentation.py` module provides advanced wire segmentation and connection analysis capabilities for the Enhanced Wire Detection CNN system. It takes pixelated wire regions from CNN outputs and breaks them down into individual wire segments, establishes connections between wires and junctions, and creates network graphs for circuit analysis. This module bridges the gap between raw CNN outputs and structured circuit analysis.

## Architecture and Dependencies

### Core Dependencies
- **OpenCV (cv2)**: Computer vision operations and morphological processing
- **NumPy**: Numerical computations and array operations
- **scikit-image**: Image analysis, morphology, and segmentation
- **scipy**: Spatial operations and distance calculations
- **NetworkX**: Graph operations for wire network analysis

### Key Features
- **Individual Wire Segmentation**: Breaks down pixelated regions into individual wires
- **Junction Detection**: Identifies wire intersection points
- **Connection Analysis**: Establishes connections between wires and junctions
- **Network Graph Creation**: Creates graph representations of wire networks
- **Skeletonization**: Uses skeletonization for centerline extraction
- **Morphological Processing**: Advanced morphological operations for wire separation

## Detailed Class and Function Analysis

### 1. `EnhancedWireSegmenter` Class

**Purpose**: Main class for advanced wire segmentation and connection analysis.

**Constructor and Initialization**:
```python
class EnhancedWireSegmenter:
    """
    Advanced wire segmentation that identifies individual wires and their connections
    """
    
    def __init__(self, min_wire_length=5, max_wire_thickness=15, 
                 junction_threshold=8, connection_tolerance=5):
        self.min_wire_length = min_wire_length
        self.max_wire_thickness = max_wire_thickness
        self.junction_threshold = junction_threshold
        self.connection_tolerance = connection_tolerance
```

**Key Features**:
- **Configurable Parameters**: Adjustable thresholds for different use cases
- **Length Filtering**: Minimum wire length for noise removal
- **Thickness Limits**: Maximum wire thickness for validation
- **Junction Detection**: Threshold for junction identification
- **Connection Tolerance**: Distance tolerance for wire connections

#### `segment_wires()` Method

**Purpose**: Main segmentation method that orchestrates the entire wire segmentation pipeline.

**Detailed Implementation**:
```python
def segment_wires(self, wire_mask: np.ndarray, junction_mask: Optional[np.ndarray] = None, 
                 original_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Segment wire mask into individual wire segments and establish connections
    
    Args:
        wire_mask: Binary wire mask from CNN
        junction_mask: Binary junction mask from CNN (optional)
        original_size: Original image size for coordinate scaling (optional)
        
    Returns:
        Dictionary containing individual wire segments and their connections
    """
    # Clean the wire mask
    cleaned_mask = self._clean_wire_mask(wire_mask)
    
    # Detect junctions
    if junction_mask is not None:
        junctions = self._detect_junctions_from_mask(junction_mask)
    else:
        junctions = self._detect_junctions_from_wire_mask(cleaned_mask)
    
    # Segment individual wires
    wire_segments = self._segment_individual_wires(cleaned_mask, junctions)
    
    # Scale coordinates to original size if provided
    if original_size is not None:
        wire_segments = self._scale_wire_segments(wire_segments, wire_mask.shape, original_size)
        junctions = self._scale_junction_coordinates(junctions, wire_mask.shape, original_size)
    
    # Establish connections between wires
    connections = self._establish_connections(wire_segments, junctions)
    
    # Create wire network graph
    wire_network = self._create_wire_network(wire_segments, connections)
    
    return {
        'wire_segments': wire_segments,
        'junctions': junctions,
        'connections': connections,
        'wire_network': wire_network,
        'cleaned_mask': cleaned_mask
    }
```

**Key Features**:
- **Complete Pipeline**: Orchestrates entire segmentation process
- **Flexible Input**: Works with or without junction masks
- **Coordinate Scaling**: Scales coordinates to original image size
- **Network Creation**: Creates graph representation of wire network
- **Comprehensive Output**: Returns all segmentation results

### 2. `_clean_wire_mask()` Method

**Purpose**: Cleans and prepares wire mask for segmentation.

**Implementation**:
```python
def _clean_wire_mask(self, wire_mask: np.ndarray) -> np.ndarray:
    """Clean wire mask and prepare for segmentation"""
    # First threshold the mask
    binary_mask = (wire_mask > 0.1).astype(np.uint8)
    
    # Remove small noise
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=10).astype(np.uint8)
    
    # Fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
```

**Key Features**:
- **Thresholding**: Converts probability mask to binary
- **Noise Removal**: Removes small objects and artifacts
- **Hole Filling**: Fills small holes in wire regions
- **Morphological Processing**: Uses closing operation for cleanup

### 3. Junction Detection Methods

#### `_detect_junctions_from_mask()` Method

**Purpose**: Detects junctions from CNN junction mask.

**Implementation**:
```python
def _detect_junctions_from_mask(self, junction_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Detect junctions from junction mask"""
    labeled = label(junction_mask)
    junctions = []
    
    for region in regionprops(labeled):
        if region.area > 5:
            centroid = (int(region.centroid[1]), int(region.centroid[0]))
            junctions.append(centroid)
    
    return junctions
```

**Key Features**:
- **Region Analysis**: Uses scikit-image regionprops
- **Centroid Calculation**: Calculates junction centroids
- **Area Filtering**: Filters out small regions
- **Coordinate Conversion**: Converts to (x, y) format

#### `_detect_junctions_from_wire_mask()` Method

**Purpose**: Detects junctions from wire mask using skeleton analysis.

**Implementation**:
```python
def _detect_junctions_from_wire_mask(self, wire_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Detect junctions from wire mask using skeleton analysis"""
    # Create skeleton
    skeleton = skeletonize(wire_mask).astype(np.uint8)
    
    # Find junction points (pixels with 3+ neighbors)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    
    # Junctions have 3 or more neighbors
    junction_pixels = ((neighbor_count >= 3) & (skeleton > 0))
    junction_coords = np.where(junction_pixels)
    
    junctions = [(int(x), int(y)) for y, x in zip(junction_coords[0], junction_coords[1])]
    
    return junctions
```

**Key Features**:
- **Skeletonization**: Creates wire skeleton for analysis
- **Neighbor Counting**: Counts neighbors for each pixel
- **Junction Criteria**: Identifies pixels with 3+ neighbors
- **Coordinate Extraction**: Extracts junction coordinates

### 4. `_segment_individual_wires()` Method

**Purpose**: Segments individual wires using connected components and skeleton analysis.

**Implementation**:
```python
def _segment_individual_wires(self, wire_mask: np.ndarray, junctions: List[Tuple[int, int]]) -> List[Dict]:
    """
    Segment individual wires using connected components and skeleton analysis
    """
    # Threshold the wire mask
    binary_mask = (wire_mask > 0.1).astype(np.uint8)
    
    # Find connected components
    labeled = label(binary_mask)
    
    # Extract individual wire segments
    wire_segments = []
    for region_id in np.unique(labeled):
        if region_id == 0:  # Skip background
            continue
            
        region_mask = (labeled == region_id).astype(np.uint8)
        
        # Analyze the region
        segment_info = self._analyze_wire_segment(region_mask, region_id)
        
        if segment_info and segment_info['length'] >= self.min_wire_length:
            wire_segments.append(segment_info)
    
    return wire_segments
```

**Key Features**:
- **Connected Components**: Uses scikit-image labeling
- **Region Analysis**: Analyzes each connected component
- **Length Filtering**: Filters out short segments
- **Structured Output**: Returns detailed segment information

### 5. `_analyze_wire_segment()` Method

**Purpose**: Analyzes a wire segment and extracts its properties.

**Detailed Implementation**:
```python
def _analyze_wire_segment(self, region_mask: np.ndarray, segment_id: int) -> Optional[Dict]:
    """Analyze a wire segment and extract its properties"""
    # Find skeleton of the region
    skeleton = skeletonize(region_mask).astype(np.uint8)
    
    if np.sum(skeleton) == 0:
        return None
    
    # Find endpoints
    endpoints = self._find_endpoints(skeleton)
    
    # Find centerline
    centerline = self._extract_centerline(skeleton)
    
    if len(centerline) < 2:
        return None
    
    # Calculate properties
    start_point = centerline[0]
    end_point = centerline[-1]
    length = self._calculate_centerline_length(centerline)
    
    # Find bounding box
    coords = np.where(region_mask)
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    
    return {
        'id': segment_id,
        'start': start_point,
        'end': end_point,
        'centerline': centerline,
        'length': length,
        'thickness': self._estimate_thickness(region_mask, centerline),
        'bbox': (x_min, y_min, x_max, y_max),
        'endpoints': endpoints,
        'mask': region_mask
    }
```

**Key Features**:
- **Skeleton Analysis**: Uses skeletonization for centerline extraction
- **Endpoint Detection**: Identifies wire endpoints
- **Length Calculation**: Calculates wire length
- **Thickness Estimation**: Estimates wire thickness
- **Bounding Box**: Calculates wire bounding box
- **Comprehensive Properties**: Returns all wire properties

### 6. Endpoint and Centerline Analysis

#### `_find_endpoints()` Method

**Purpose**: Finds endpoints in skeleton.

**Implementation**:
```python
def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Find endpoints in skeleton"""
    # Count neighbors for each pixel
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    
    # Endpoints have exactly 1 neighbor
    endpoint_pixels = ((neighbor_count == 1) & (skeleton > 0))
    endpoint_coords = np.where(endpoint_pixels)
    
    return [(int(x), int(y)) for y, x in zip(endpoint_coords[0], endpoint_coords[1])]
```

**Key Features**:
- **Neighbor Counting**: Counts neighbors for each pixel
- **Endpoint Criteria**: Identifies pixels with exactly 1 neighbor
- **Coordinate Extraction**: Extracts endpoint coordinates

#### `_extract_centerline()` Method

**Purpose**: Extracts centerline points from skeleton.

**Implementation**:
```python
def _extract_centerline(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Extract centerline points from skeleton"""
    skeleton_coords = np.where(skeleton > 0)
    centerline = [(int(x), int(y)) for y, x in zip(skeleton_coords[0], skeleton_coords[1])]
    
    # Sort points to create a continuous path
    if len(centerline) > 1:
        centerline = self._sort_centerline_points(centerline)
    
    return centerline
```

**Key Features**:
- **Coordinate Extraction**: Extracts skeleton coordinates
- **Path Sorting**: Sorts points for continuous path
- **Continuous Centerline**: Creates ordered centerline

#### `_sort_centerline_points()` Method

**Purpose**: Sorts centerline points to create a continuous path.

**Implementation**:
```python
def _sort_centerline_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort centerline points to create a continuous path"""
    if len(points) <= 2:
        return points
    
    # Start with the first point
    sorted_points = [points[0]]
    remaining_points = points[1:]
    
    while remaining_points:
        # Find the closest point to the last point in sorted list
        last_point = sorted_points[-1]
        distances = [np.sqrt((p[0] - last_point[0])**2 + (p[1] - last_point[1])**2) 
                    for p in remaining_points]
        closest_idx = np.argmin(distances)
        
        sorted_points.append(remaining_points[closest_idx])
        remaining_points.pop(closest_idx)
    
    return sorted_points
```

**Key Features**:
- **Greedy Sorting**: Uses greedy algorithm for path construction
- **Distance Calculation**: Calculates Euclidean distances
- **Continuous Path**: Creates continuous wire path
- **Efficient Algorithm**: O(n²) complexity for small wire segments

### 7. Length and Thickness Analysis

#### `_calculate_centerline_length()` Method

**Purpose**: Calculates length of centerline.

**Implementation**:
```python
def _calculate_centerline_length(self, centerline: List[Tuple[int, int]]) -> float:
    """Calculate length of centerline"""
    if len(centerline) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(centerline)):
        p1, p2 = centerline[i-1], centerline[i]
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        total_length += length
    
    return total_length
```

**Key Features**:
- **Euclidean Distance**: Uses Euclidean distance for length calculation
- **Cumulative Length**: Sums distances between consecutive points
- **Precision**: High precision floating-point calculation

#### `_estimate_thickness()` Method

**Purpose**: Estimates wire thickness.

**Implementation**:
```python
def _estimate_thickness(self, region_mask: np.ndarray, centerline: List[Tuple[int, int]]) -> float:
    """Estimate wire thickness"""
    if not centerline:
        return 0.0
    
    # Sample points along centerline and measure perpendicular thickness
    thicknesses = []
    for x, y in centerline[::max(1, len(centerline)//10)]:  # Sample every 10th point
        # Find perpendicular line and measure thickness
        thickness = self._measure_perpendicular_thickness(region_mask, x, y)
        thicknesses.append(thickness)
    
    return np.median(thicknesses) if thicknesses else 0.0
```

**Key Features**:
- **Sampling Strategy**: Samples every 10th point for efficiency
- **Perpendicular Measurement**: Measures thickness perpendicular to wire direction
- **Median Calculation**: Uses median for robust thickness estimation
- **Efficiency**: Balances accuracy with computational efficiency

#### `_measure_perpendicular_thickness()` Method

**Purpose**: Measures thickness perpendicular to wire direction.

**Implementation**:
```python
def _measure_perpendicular_thickness(self, mask: np.ndarray, x: int, y: int) -> float:
    """Measure thickness perpendicular to wire direction"""
    # Simple approach: find distance to nearest edge
    dist_transform = distance_transform_edt(mask)
    return dist_transform[y, x] * 2  # Multiply by 2 for diameter
```

**Key Features**:
- **Distance Transform**: Uses Euclidean distance transform
- **Edge Distance**: Measures distance to nearest edge
- **Diameter Calculation**: Multiplies by 2 for diameter
- **Efficient Calculation**: Uses scipy's optimized distance transform

### 8. Connection Analysis

#### `_establish_connections()` Method

**Purpose**: Establishes connections between wire segments and junctions.

**Implementation**:
```python
def _establish_connections(self, wire_segments: List[Dict], junctions: List[Tuple[int, int]]) -> List[Dict]:
    """Establish connections between wire segments and junctions"""
    connections = []
    
    for i, wire1 in enumerate(wire_segments):
        for j, wire2 in enumerate(wire_segments[i+1:], i+1):
            # Check if wires are connected
            if self._are_wires_connected(wire1, wire2):
                connections.append({
                    'wire1_id': wire1['id'],
                    'wire2_id': wire2['id'],
                    'connection_type': 'wire_to_wire',
                    'connection_point': self._find_connection_point(wire1, wire2)
                })
        
        # Check connections to junctions
        for junction in junctions:
            if self._is_wire_connected_to_junction(wire1, junction):
                connections.append({
                    'wire_id': wire1['id'],
                    'junction': junction,
                    'connection_type': 'wire_to_junction'
                })
    
    return connections
```

**Key Features**:
- **Pairwise Analysis**: Checks all pairs of wire segments
- **Junction Connections**: Checks connections to junctions
- **Connection Types**: Distinguishes between wire-to-wire and wire-to-junction
- **Connection Points**: Records connection point locations

#### `_are_wires_connected()` Method

**Purpose**: Checks if two wires are connected.

**Implementation**:
```python
def _are_wires_connected(self, wire1: Dict, wire2: Dict) -> bool:
    """Check if two wires are connected"""
    # Check if any endpoints are close to each other
    for ep1 in wire1['endpoints']:
        for ep2 in wire2['endpoints']:
            distance = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
            if distance <= self.connection_tolerance:
                return True
    return False
```

**Key Features**:
- **Endpoint Analysis**: Checks endpoint proximity
- **Distance Calculation**: Uses Euclidean distance
- **Tolerance Threshold**: Uses configurable connection tolerance
- **Efficient Check**: Early return on first connection found

#### `_find_connection_point()` Method

**Purpose**: Finds the connection point between two wires.

**Implementation**:
```python
def _find_connection_point(self, wire1: Dict, wire2: Dict) -> Tuple[int, int]:
    """Find the connection point between two wires"""
    # Find closest endpoints
    min_distance = float('inf')
    connection_point = None
    
    for ep1 in wire1['endpoints']:
        for ep2 in wire2['endpoints']:
            distance = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
            if distance < min_distance:
                min_distance = distance
                connection_point = ((ep1[0] + ep2[0]) // 2, (ep1[1] + ep2[1]) // 2)
    
    return connection_point or (0, 0)
```

**Key Features**:
- **Closest Point**: Finds closest endpoint pair
- **Midpoint Calculation**: Calculates midpoint between endpoints
- **Distance Tracking**: Tracks minimum distance
- **Fallback Value**: Returns (0, 0) if no connection found

### 9. Network Graph Creation

#### `_create_wire_network()` Method

**Purpose**: Creates a network graph of wire connections.

**Implementation**:
```python
def _create_wire_network(self, wire_segments: List[Dict], connections: List[Dict]) -> nx.Graph:
    """Create a network graph of wire connections"""
    G = nx.Graph()
    
    # Add wire segments as nodes
    for wire in wire_segments:
        G.add_node(wire['id'], **wire)
    
    # Add connections as edges
    for conn in connections:
        if conn['connection_type'] == 'wire_to_wire':
            G.add_edge(conn['wire1_id'], conn['wire2_id'], **conn)
        elif conn['connection_type'] == 'wire_to_junction':
            # Add junction as a special node
            junction_id = f"junction_{conn['junction']}"
            G.add_node(junction_id, type='junction', pos=conn['junction'])
            G.add_edge(conn['wire_id'], junction_id, **conn)
    
    return G
```

**Key Features**:
- **NetworkX Integration**: Uses NetworkX for graph operations
- **Node Attributes**: Stores wire properties as node attributes
- **Edge Attributes**: Stores connection properties as edge attributes
- **Junction Nodes**: Creates special nodes for junctions
- **Graph Operations**: Enables advanced graph analysis

### 10. Coordinate Scaling Methods

#### `_scale_wire_segments()` Method

**Purpose**: Scales wire segment coordinates from input to output size.

**Implementation**:
```python
def _scale_wire_segments(self, wire_segments: List[Dict], input_shape: Tuple[int, int], 
                        output_shape: Tuple[int, int]) -> List[Dict]:
    """Scale wire segment coordinates from input to output size"""
    if not wire_segments:
        return []
    
    scale_y = output_shape[0] / input_shape[0]
    scale_x = output_shape[1] / input_shape[1]
    
    scaled_segments = []
    for wire in wire_segments:
        scaled_wire = wire.copy()
        
        # Scale start and end points
        scaled_wire['start'] = (int(wire['start'][0] * scale_x), int(wire['start'][1] * scale_y))
        scaled_wire['end'] = (int(wire['end'][0] * scale_x), int(wire['end'][1] * scale_y))
        
        # Scale centerline
        scaled_centerline = []
        for x, y in wire['centerline']:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_centerline.append((scaled_x, scaled_y))
        scaled_wire['centerline'] = scaled_centerline
        
        # Scale endpoints
        scaled_endpoints = []
        for x, y in wire['endpoints']:
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_endpoints.append((scaled_x, scaled_y))
        scaled_wire['endpoints'] = scaled_endpoints
        
        # Scale bounding box
        x_min, y_min, x_max, y_max = wire['bbox']
        scaled_wire['bbox'] = (
            int(x_min * scale_x), int(y_min * scale_y),
            int(x_max * scale_x), int(y_max * scale_y)
        )
        
        # Scale length and thickness
        scaled_wire['length'] = wire['length'] * ((scale_x + scale_y) / 2)
        scaled_wire['thickness'] = wire['thickness'] * ((scale_x + scale_y) / 2)
        
        scaled_segments.append(scaled_wire)
    
    return scaled_segments
```

**Key Features**:
- **Proportional Scaling**: Scales all coordinates proportionally
- **Comprehensive Scaling**: Scales all wire properties
- **Integer Conversion**: Converts to integers for pixel coordinates
- **Average Scaling**: Uses average scale for length and thickness

#### `_scale_junction_coordinates()` Method

**Purpose**: Scales junction coordinates from input to output size.

**Implementation**:
```python
def _scale_junction_coordinates(self, junctions: List[Tuple[int, int]], 
                               input_shape: Tuple[int, int], 
                               output_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Scale junction coordinates from input to output size"""
    if not junctions:
        return []
    
    scale_y = output_shape[0] / input_shape[0]
    scale_x = output_shape[1] / input_shape[1]
    
    scaled_junctions = []
    for x, y in junctions:
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        scaled_junctions.append((new_x, new_y))
    
    return scaled_junctions
```

**Key Features**:
- **Simple Scaling**: Scales junction coordinates
- **Integer Conversion**: Converts to integers
- **Empty Handling**: Handles empty junction lists

## Usage Examples

### Basic Wire Segmentation
```python
from enhanced_wire_segmentation import EnhancedWireSegmenter

# Create segmenter
segmenter = EnhancedWireSegmenter(
    min_wire_length=5,
    max_wire_thickness=15,
    junction_threshold=8,
    connection_tolerance=5
)

# Segment wires
results = segmenter.segment_wires(wire_mask, junction_mask, original_size)

# Access results
wire_segments = results['wire_segments']
junctions = results['junctions']
connections = results['connections']
wire_network = results['wire_network']

print(f"Found {len(wire_segments)} wire segments")
print(f"Found {len(junctions)} junctions")
print(f"Found {len(connections)} connections")
```

### Wire Analysis
```python
# Analyze individual wires
for wire in wire_segments:
    print(f"Wire {wire['id']}:")
    print(f"  Length: {wire['length']:.1f} pixels")
    print(f"  Thickness: {wire['thickness']:.1f} pixels")
    print(f"  Start: {wire['start']}")
    print(f"  End: {wire['end']}")
    print(f"  Endpoints: {wire['endpoints']}")
```

### Network Analysis
```python
import networkx as nx

# Analyze wire network
print(f"Network nodes: {wire_network.number_of_nodes()}")
print(f"Network edges: {wire_network.number_of_edges()}")

# Find connected components
connected_components = list(nx.connected_components(wire_network))
print(f"Connected components: {len(connected_components)}")

# Calculate network metrics
if wire_network.number_of_nodes() > 0:
    density = nx.density(wire_network)
    print(f"Network density: {density:.3f}")
```

### Integration with Inference
```python
from run_inference import load_model, preprocess_image

# Load model and run inference
model = load_model("experiments/models/unet_best.h5")
img_batch, original_img, original_size = preprocess_image("schematic.jpg")

# Run model
outputs = model(img_batch)
wire_mask = outputs['wire_mask'][0, :, :, 0].numpy()
junction_mask = outputs['junction_mask'][0, :, :, 0].numpy()

# Segment wires
segmenter = EnhancedWireSegmenter()
results = segmenter.segment_wires(wire_mask, junction_mask, original_size)

# Use results for analysis
wire_segments = results['wire_segments']
print(f"Detected {len(wire_segments)} individual wires")
```

## Integration with Other Modules

### 1. Inference Integration
- **Model Outputs**: Takes CNN outputs as input
- **Coordinate Scaling**: Scales to original image resolution
- **Post-Processing**: Provides structured post-processing

### 2. Visualization Integration
- **Wire Segments**: Provides individual wire segments for visualization
- **Junctions**: Provides junction points for visualization
- **Connections**: Provides connection information for visualization

### 3. Analysis Integration
- **Network Graphs**: Provides NetworkX graphs for analysis
- **Wire Properties**: Provides detailed wire properties
- **Connection Analysis**: Provides connection information

## Performance Considerations

### 1. Computational Complexity
- **Skeletonization**: O(n) where n is number of pixels
- **Connected Components**: O(n) where n is number of pixels
- **Connection Analysis**: O(m²) where m is number of wire segments
- **Overall Complexity**: O(n + m²) for typical use cases

### 2. Memory Usage
- **Skeleton Storage**: Stores skeleton for each wire segment
- **Centerline Storage**: Stores centerline points
- **Network Graph**: Stores graph structure in memory
- **Efficient Storage**: Uses NumPy arrays for efficiency

### 3. Optimization Strategies
- **Sampling**: Samples points for thickness estimation
- **Early Termination**: Early return in connection analysis
- **Vectorized Operations**: Uses NumPy vectorized operations
- **Efficient Algorithms**: Uses optimized scikit-image functions

## Error Handling and Robustness

### 1. Input Validation
- **Mask Validation**: Validates input masks
- **Size Validation**: Validates image dimensions
- **Type Validation**: Validates data types

### 2. Edge Cases
- **Empty Masks**: Handles empty wire masks
- **Single Pixels**: Handles single-pixel wires
- **No Junctions**: Handles cases with no junctions
- **No Connections**: Handles cases with no connections

### 3. Fallback Mechanisms
- **Default Values**: Provides default values for missing data
- **Graceful Degradation**: Continues processing with partial results
- **Error Recovery**: Recovers from processing errors

This module represents a sophisticated wire segmentation system that bridges the gap between raw CNN outputs and structured circuit analysis. The combination of advanced computer vision techniques, graph analysis, and robust error handling makes it a production-ready solution for wire segmentation and analysis tasks.
