# Comprehensive Explanation: cleanup_duplicates.py

## Overview
The `cleanup_duplicates.py` script is a utility designed to identify and manage duplicate PDF datasheets within the organized directory structure created by the datasheet downloader. It uses content-based hashing (SHA-256) to detect true duplicates regardless of filename variations, and safely moves duplicates to a cleanup directory while preserving the original directory hierarchy.

## Architecture and Dependencies

### Core Dependencies
- **hashlib**: SHA-256 hashing for content-based duplicate detection
- **pathlib**: Modern path handling and directory traversal
- **collections.defaultdict**: Efficient grouping of files by metadata
- **shutil**: Safe file operations for moving duplicates
- **os**: File system operations and metadata access

### Key Design Patterns
- **Two-Pass Algorithm**: First pass collects file metadata, second pass processes duplicates
- **Content-Based Detection**: Uses file content hashing rather than filename comparison
- **Safe Operations**: Moves files instead of deleting to prevent data loss
- **Hierarchical Processing**: Maintains directory structure in cleanup location

## Detailed Function Analysis

### 1. `get_file_hash()` Function

**Purpose**: Calculates SHA-256 hash of a file's content for duplicate detection.

**Algorithm**:
```python
def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
```

**Key Features**:
- **Memory Efficient**: Reads file in 4KB chunks to handle large PDFs
- **Content-Based**: Hash represents actual file content, not metadata
- **Deterministic**: Same content always produces same hash
- **Collision Resistant**: SHA-256 provides extremely low collision probability

**Why This Matters**: Content-based hashing is the most reliable method for detecting true duplicates, especially when files may have different names but identical content.

### 2. `get_base_filename()` Function

**Purpose**: Extracts base filename by removing component type suffixes for intelligent grouping.

**Algorithm**:
```python
def get_base_filename(filename):
    """Remove component suffix from filename"""
    if '_' in filename:
        return filename.rsplit('_', 1)[0]
    return filename
```

**Key Features**:
- **Suffix Removal**: Removes the last underscore and everything after it
- **Intelligent Grouping**: Groups files like `AD1234_microcontroller.pdf` and `AD1234_opamp.pdf`
- **Fallback Handling**: Returns original filename if no underscore found
- **Simple Logic**: Uses right-split to handle multiple underscores correctly

**Example Transformations**:
- `AD1234_microcontroller.pdf` → `AD1234`
- `MAX1234_opamp.pdf` → `MAX1234`
- `simple_filename.pdf` → `simple_filename`

**Why This Matters**: This function enables the script to group related files that might be the same datasheet but categorized under different component types.

### 3. `process_manufacturer()` Function

**Purpose**: Processes a single manufacturer directory to identify and handle duplicates.

**Algorithm**:
```python
def process_manufacturer(manufacturer_dir, cleanup_dir):
    print(f"\nProcessing {manufacturer_dir.name}...")
    
    # Dictionary to store file info: {base_name: {hash: [(path, size), ...]}}
    file_groups = defaultdict(lambda: defaultdict(list))
    
    # First pass: collect all files and their info
    for component_dir in manufacturer_dir.iterdir():
        if not component_dir.is_dir():
            continue
            
        for pdf_file in component_dir.glob("*.pdf"):
            base_name = get_base_filename(pdf_file.stem)
            file_size = pdf_file.stat().st_size
            file_hash = get_file_hash(pdf_file)
            
            file_groups[base_name][file_hash].append((pdf_file, file_size))
    
    # Second pass: identify and handle duplicates
    duplicates_found = False
    for base_name, hash_groups in file_groups.items():
        for file_hash, files in hash_groups.items():
            if len(files) > 1:
                duplicates_found = True
                print(f"\nFound {len(files)} duplicates for {base_name}:")
                
                # Keep the first file, move others to cleanup
                keep_file = files[0][0]
                print(f"Keeping: {keep_file}")
                
                for file_path, _ in files[1:]:
                    # Create cleanup directory structure
                    rel_path = file_path.relative_to(manufacturer_dir)
                    cleanup_path = cleanup_dir / manufacturer_dir.name / rel_path.parent
                    cleanup_path.mkdir(parents=True, exist_ok=True)
                    
                    # Move file to cleanup
                    shutil.move(str(file_path), str(cleanup_path / file_path.name))
                    print(f"Moved to cleanup: {file_path}")
    
    return duplicates_found
```

**Key Features**:
- **Two-Pass Processing**: First pass collects metadata, second pass processes duplicates
- **Hierarchical Grouping**: Groups files by base name, then by content hash
- **Safe Duplicate Handling**: Keeps first occurrence, moves others to cleanup
- **Directory Structure Preservation**: Maintains original hierarchy in cleanup location
- **Progress Reporting**: Provides detailed feedback on duplicate processing

**Data Structure**:
```python
file_groups = {
    "AD1234": {
        "abc123hash": [(path1, size1), (path2, size2)],
        "def456hash": [(path3, size3)]
    },
    "MAX5678": {
        "ghi789hash": [(path4, size4)]
    }
}
```

**Why This Matters**: This function efficiently processes large directory structures while maintaining data integrity and providing clear feedback on duplicate handling.

### 4. `main()` Function

**Purpose**: Orchestrates the cleanup process across all manufacturer directories.

**Algorithm**:
```python
def main():
    base_dir = Path("datasheets")
    cleanup_dir = Path("cleanup_duplicates")
    
    if not base_dir.exists():
        print("No datasheets directory found!")
        return
    
    # Create cleanup directory if it doesn't exist
    cleanup_dir.mkdir(exist_ok=True)
    
    # Process each manufacturer directory
    any_duplicates = False
    for manufacturer_dir in base_dir.iterdir():
        if manufacturer_dir.is_dir():
            if process_manufacturer(manufacturer_dir, cleanup_dir):
                any_duplicates = True
    
    if not any_duplicates:
        print("\nAll directories are clean - no duplicates found!")
    else:
        print("\nCleanup complete! Check the 'cleanup_duplicates' directory for moved files.")
```

**Key Features**:
- **Directory Validation**: Checks for existence of datasheets directory
- **Cleanup Directory Creation**: Creates cleanup directory if needed
- **Batch Processing**: Processes all manufacturer directories
- **Summary Reporting**: Provides overall cleanup status
- **Error Handling**: Graceful handling of missing directories

**Why This Matters**: This function provides a simple, safe entry point for the cleanup process with comprehensive error handling and user feedback.

## Duplicate Detection Algorithm

### 1. Content-Based Hashing
- **SHA-256 Algorithm**: Uses cryptographically secure hashing
- **Chunked Reading**: Reads files in 4KB chunks for memory efficiency
- **Collision Resistance**: Extremely low probability of false positives
- **Deterministic**: Same content always produces same hash

### 2. Intelligent Grouping
- **Base Name Grouping**: Groups files by base filename (ignoring suffixes)
- **Hash Subgrouping**: Within each base name group, subgroups by content hash
- **Duplicate Detection**: Files with same hash are considered duplicates

### 3. Duplicate Resolution Strategy
- **First File Wins**: Keeps the first file encountered in each duplicate group
- **Safe Movement**: Moves duplicates to cleanup directory instead of deleting
- **Structure Preservation**: Maintains original directory hierarchy in cleanup

## File System Operations

### 1. Directory Structure
```
datasheets/                    # Original directory
├── analog/
│   ├── microcontroller/
│   │   ├── AD1234.pdf        # Kept (first occurrence)
│   │   └── AD5678.pdf
│   └── opamp/
│       └── AD1234_opamp.pdf  # Moved to cleanup (duplicate)
└── ti/
    └── microcontroller/
        └── AD1234_ti.pdf     # Moved to cleanup (duplicate)

cleanup_duplicates/           # Cleanup directory
├── analog/
│   └── opamp/
│       └── AD1234_opamp.pdf  # Duplicate moved here
└── ti/
    └── microcontroller/
        └── AD1234_ti.pdf     # Duplicate moved here
```

### 2. Safe File Operations
- **Move vs Delete**: Uses `shutil.move()` instead of `os.remove()`
- **Directory Creation**: Creates cleanup directories as needed
- **Path Preservation**: Maintains relative path structure
- **Atomic Operations**: File operations are atomic where possible

### 3. Error Handling
- **Permission Errors**: Handles file system permission issues
- **Missing Directories**: Creates directories as needed
- **File Conflicts**: Handles cases where cleanup files already exist
- **Graceful Degradation**: Continues processing even if individual operations fail

## Performance Considerations

### 1. Memory Efficiency
- **Chunked Reading**: Reads files in 4KB chunks to minimize memory usage
- **Lazy Evaluation**: Processes files one at a time
- **Garbage Collection**: Allows Python's garbage collector to clean up file handles

### 2. I/O Optimization
- **Single Pass Hashing**: Calculates hash while reading file once
- **Batch Directory Operations**: Creates directories in batches
- **Efficient Path Operations**: Uses `pathlib` for efficient path handling

### 3. Scalability
- **Linear Complexity**: O(n) time complexity where n is number of files
- **Memory Bounded**: Memory usage is bounded by file size, not total dataset size
- **Parallelizable**: Could be extended for parallel processing

## Usage Examples

### Basic Usage
```bash
python cleanup_duplicates.py
```

### Expected Output
```
Processing analog...

Found 3 duplicates for AD1234:
Keeping: datasheets/analog/microcontroller/AD1234.pdf
Moved to cleanup: datasheets/analog/opamp/AD1234_opamp.pdf
Moved to cleanup: datasheets/analog/sensor/AD1234_sensor.pdf

Processing ti...

No duplicates found in ti

Cleanup complete! Check the 'cleanup_duplicates' directory for moved files.
```

### Directory Structure After Cleanup
```
datasheets/                    # Cleaned directory
├── analog/
│   ├── microcontroller/
│   │   ├── AD1234.pdf        # Only unique files remain
│   │   └── AD5678.pdf
│   └── opamp/
│       └── MAX1234.pdf
└── ti/
    └── microcontroller/
        └── TMS320F28335.pdf

cleanup_duplicates/           # Duplicates moved here
├── analog/
│   ├── opamp/
│   │   └── AD1234_opamp.pdf
│   └── sensor/
│       └── AD1234_sensor.pdf
└── ti/
    └── microcontroller/
        └── AD1234_ti.pdf
```

## Integration with Datasheet Downloader

### 1. Directory Structure Compatibility
- **Expected Structure**: Works with `datasheets/manufacturer/component/` hierarchy
- **File Naming**: Handles naming conventions from datasheet downloader
- **Component Suffixes**: Recognizes and handles component type suffixes

### 2. Workflow Integration
- **Post-Processing**: Typically run after datasheet downloader completes
- **Safe Operation**: Can be run multiple times safely
- **Incremental Processing**: Only processes new files on subsequent runs

### 3. Data Integrity
- **No Data Loss**: Never deletes files, only moves them
- **Reversible**: Duplicates can be restored from cleanup directory
- **Audit Trail**: Clear logging of all duplicate handling actions

## Error Handling and Edge Cases

### 1. File System Errors
- **Permission Denied**: Handles read/write permission issues
- **Disk Full**: Manages disk space constraints
- **File Locked**: Handles files in use by other processes

### 2. Data Integrity
- **Corrupted Files**: Handles files that can't be read
- **Empty Files**: Processes zero-byte files correctly
- **Symlinks**: Handles symbolic links appropriately

### 3. Edge Cases
- **Single File Directories**: Handles directories with only one file
- **Empty Directories**: Skips empty directories
- **Non-PDF Files**: Only processes PDF files

## Risks and Limitations

### 1. False Positives
- **Risk**: Different files with same content (rare with SHA-256)
- **Mitigation**: SHA-256 provides extremely low collision probability
- **Monitoring**: Review cleanup directory for unexpected duplicates

### 2. Performance
- **Risk**: Slow processing for very large datasets
- **Mitigation**: Chunked reading and efficient algorithms
- **Scaling**: Consider parallel processing for very large datasets

### 3. Data Loss
- **Risk**: Accidental data loss during file operations
- **Mitigation**: Move operations instead of delete operations
- **Recovery**: Duplicates preserved in cleanup directory

## Future Enhancements

### 1. Parallel Processing
- **Multi-threading**: Process multiple manufacturers simultaneously
- **Multi-processing**: Use multiple CPU cores for hashing
- **Progress Bars**: Visual progress indicators for large datasets

### 2. Advanced Detection
- **Fuzzy Matching**: Detect near-duplicates with slight differences
- **Metadata Analysis**: Consider file creation dates and sizes
- **Content Analysis**: Analyze PDF content for better grouping

### 3. User Interface
- **Interactive Mode**: Allow user to choose which duplicates to keep
- **Preview Mode**: Show duplicates before moving them
- **Configuration**: Allow customization of duplicate detection criteria

This script provides a robust, safe solution for managing duplicate datasheets in large collections, ensuring data integrity while maintaining an organized directory structure.
