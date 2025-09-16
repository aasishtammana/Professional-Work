# Comprehensive Explanation: datasheet_downloader.py

## Overview
The `datasheet_downloader.py` script is the core component of the Google Query Datasheet Retriever system. It automatically searches for and downloads PDF datasheets from various electronic component manufacturers using Google search queries and Selenium WebDriver automation. The script implements sophisticated anti-detection measures and organizes downloads in a structured directory hierarchy.

**Note**: This functionality is also available as an interactive Jupyter notebook (`Google query datasheet downloader.ipynb`) which provides the same core functionality with enhanced usability, real-time progress monitoring, and easy configuration modification through cell-by-cell execution.

## Architecture and Dependencies

### Core Dependencies
- **Selenium WebDriver**: Browser automation for handling dynamic content and CAPTCHAs
- **requests**: HTTP requests for basic web interactions
- **BeautifulSoup**: HTML parsing and content extraction
- **pandas**: Data manipulation and analysis
- **webdriver-manager**: Automatic ChromeDriver management
- **urllib.parse**: URL parsing and manipulation
- **hashlib**: File integrity verification
- **concurrent.futures**: Parallel processing capabilities

### Key Design Patterns
- **Singleton Browser Instance**: Reuses single WebDriver instance across operations
- **Factory Pattern**: Generates search queries and filenames based on manufacturer patterns
- **Strategy Pattern**: Different handling strategies for different manufacturers
- **Observer Pattern**: Monitors download progress and file system changes

## Detailed Function Analysis

### 1. `DatasheetDownloader` Class

**Purpose**: Main orchestrator class that manages the entire datasheet discovery and download process.

**Key Attributes**:
- `base_download_dir`: Root directory for all downloads (default: "datasheets")
- `user_agents`: List of rotating user agents for anti-detection
- `manufacturer_patterns`: URL patterns specific to each manufacturer's datasheet structure
- `driver`: Singleton Selenium WebDriver instance

**Initialization Logic**:
```python
def __init__(self, base_download_dir="datasheets"):
    self.base_download_dir = base_download_dir
    self.user_agents = [/* 5 different user agent strings */]
    self.manufacturer_patterns = {
        "renesas.com": "/dst/",
        "ti.com": "/ds/",
        "analog.com": "/data-sheets/",
        "st.com": "/datasheet/",
        "microchip.com": "/en/",
        /* ... 10+ manufacturer patterns */
    }
    self.setup_directories()
    self.driver = None
```

**Why This Matters**: The class encapsulates all the complexity of web scraping, anti-detection, and file management. The manufacturer patterns are crucial for filtering relevant datasheet URLs from Google search results.

### 2. `generate_query()` Function

**Purpose**: Creates targeted Google search queries for specific manufacturers and component types.

**Algorithm**:
```python
def generate_query(self, manufacturer, component_type=None):
    url_pattern = self.manufacturer_patterns.get(manufacturer, "/document/")
    base_query = f"site:{manufacturer} inurl:{url_pattern} filetype:pdf"
    if component_type:
        base_query += f" {component_type}"
    return base_query
```

**Key Features**:
- **Site-specific Search**: Uses `site:` operator to limit results to manufacturer domains
- **URL Pattern Matching**: Uses `inurl:` to find datasheet-specific URL patterns
- **File Type Filtering**: Restricts results to PDF files only
- **Component Type Integration**: Adds component type keywords for more targeted results

**Example Outputs**:
- `site:ti.com inurl:/ds/ filetype:pdf microcontroller`
- `site:analog.com inurl:/data-sheets/ filetype:pdf opamp`

**Why This Matters**: These queries are highly optimized to return only relevant datasheet PDFs, reducing noise and improving download success rates.

### 3. `extract_component_info()` Function

**Purpose**: Extracts component part numbers and identifiers from URLs and titles using regex patterns.

**Algorithm**:
```python
def extract_component_info(self, url, title):
    patterns = [
        r'[A-Z]{2,3}\d{3,4}[A-Z]?',      # AD1234A, MAX1234
        r'[A-Z]\d{2,3}[A-Z]?',           # A123, B45C
        r'\d{2,3}[A-Z]\d{2,3}',          # 123A456
        r'[A-Z]\d{2,3}[A-Z]\d{2,3}',     # A123B456
        r'[A-Z]{1,4}\d{1,4}[A-Z]?\d{0,2}[A-Z]?',  # Complex patterns
        r'[A-Z]{1,4}-\d{1,4}[A-Z]?\d{0,2}[A-Z]?'   # With hyphens
    ]
    
    # Try title first, then URL, then URL path extraction
    for pattern in patterns:
        matches = re.findall(pattern, title)
        if matches:
            return matches[0]
    # ... fallback logic
```

**Key Features**:
- **Multi-pattern Matching**: Uses 6 different regex patterns to catch various component naming conventions
- **Priority-based Extraction**: Tries title first, then URL, then URL path
- **Fallback Mechanisms**: Multiple extraction strategies for edge cases
- **Manufacturer-agnostic**: Works across different component naming schemes

**Why This Matters**: Accurate component identification is crucial for proper file naming and organization. This function handles the complexity of different manufacturer naming conventions.

### 4. `generate_unique_filename()` Function

**Purpose**: Creates meaningful, unique filenames for downloaded datasheets.

**Algorithm**:
```python
def generate_unique_filename(self, url, title, manufacturer, component_type):
    if title != "Untitled":
        return self.clean_filename(title)
    
    component_number = self.extract_component_info(url, title)
    if component_number:
        return f"{component_number}_{component_type}"
    
    # URL-based fallback
    path = urlparse(url).path
    parts = [p for p in path.split('/') if p]
    if parts:
        last_part = parts[-1].split('.')[0].split('?')[0]
        if last_part and last_part != "document" and last_part != "dst":
            return f"{last_part}_{component_type}"
    
    # Timestamp fallback
    timestamp = int(time.time())
    return f"{component_type}_{timestamp}"
```

**Key Features**:
- **Title Priority**: Uses original title when available
- **Component-based Naming**: Uses extracted component numbers for meaningful names
- **URL Fallback**: Extracts meaningful parts from URL paths
- **Timestamp Safety**: Ensures uniqueness with timestamp fallback
- **Filename Sanitization**: Removes invalid characters and normalizes format

**Why This Matters**: Proper filename generation ensures organized, searchable datasheet collections and prevents file conflicts.

### 5. `setup_browser()` Function

**Purpose**: Configures and initializes a Chrome WebDriver instance with anti-detection measures.

**Detailed Configuration**:
```python
def setup_browser(self, download_dir):
    chrome_options = Options()
    
    # Basic stealth options
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={self.get_random_user_agent()}")
    
    # Anti-detection measures
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Random window size
    window_sizes = [(1920, 1080), (1366, 768), (1440, 900), (1536, 864)]
    width, height = random.choice(window_sizes)
    chrome_options.add_argument(f"--window-size={width},{height}")
    
    # Download configuration
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "safebrowsing.enabled": False
    })
```

**Key Features**:
- **Anti-Detection**: Multiple measures to avoid bot detection
- **Random User Agents**: Rotates between different browser signatures
- **Random Window Sizes**: Varies browser window dimensions
- **Download Automation**: Configures automatic PDF downloads
- **CDP Commands**: Uses Chrome DevTools Protocol to hide automation traces

**Why This Matters**: Web scraping requires sophisticated anti-detection measures to avoid IP blocking and CAPTCHA challenges. This configuration maximizes success rates.

### 6. `handle_captcha()` Function

**Purpose**: Detects and handles CAPTCHA challenges with user intervention.

**Algorithm**:
```python
def handle_captcha(self, driver, wait_time=300):
    if "unusual traffic" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
        print("\nCAPTCHA detected! Please solve it manually...")
        print("Waiting for CAPTCHA to be solved...")
        
        start_time = time.time()
        while time.time() - start_time < wait_time:
            if "unusual traffic" not in driver.page_source.lower() and "captcha" not in driver.page_source.lower():
                print("CAPTCHA solved! Continuing...")
                return True
            time.sleep(5)
        print("Timeout waiting for CAPTCHA. Moving to next search...")
        return False
    return True
```

**Key Features**:
- **Automatic Detection**: Monitors page content for CAPTCHA indicators
- **User Intervention**: Pauses execution for manual CAPTCHA solving
- **Timeout Handling**: Moves on if CAPTCHA isn't solved within timeout
- **Non-blocking**: Continues execution if no CAPTCHA is detected

**Why This Matters**: CAPTCHAs are a common anti-bot measure. This function provides a graceful way to handle them without breaking the entire download process.

### 7. `download_datasheet()` Function

**Purpose**: Downloads a single datasheet PDF with comprehensive error handling and file management.

**Detailed Workflow**:
```python
def download_datasheet(self, url, manufacturer, title, component_type=None):
    # 1. Generate filename and check for existing files
    base_filename = self.generate_unique_filename(url, title, manufacturer, component_type)
    filename = base_filename + '.pdf'
    filepath = os.path.join(self.get_manufacturer_folder(manufacturer, component_type), filename)
    
    if os.path.exists(filepath):
        print(f"Skipping {filename} - already exists")
        return True, filename
    
    # 2. Setup browser and navigate to URL
    download_dir = os.path.abspath(self.get_manufacturer_folder(manufacturer, component_type))
    driver = self.setup_browser(download_dir)
    
    # 3. Handle page loading and CAPTCHA
    driver.get(url)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    if not self.handle_captcha(driver):
        return False, None
    
    # 4. Attempt to click download buttons
    download_button_patterns = [
        "//button[contains(., 'Download')]",
        "//a[contains(., 'Download')]",
        "//button[contains(@class, 'download')]",
        "//a[contains(@class, 'download')]",
        "//button[contains(@id, 'download')]",
        "//a[contains(@id, 'download')]"
    ]
    
    # 5. Monitor for download completion
    max_wait_time = 30
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        after_files = self.get_current_pdf_files(download_dir)
        downloaded_file = self.get_new_downloaded_file(before_files, after_files)
        if downloaded_file:
            # Handle file renaming and success
            return True, filename
        time.sleep(1)
```

**Key Features**:
- **Duplicate Prevention**: Skips already downloaded files
- **Multiple Download Strategies**: Tries both automatic and button-click downloads
- **File Monitoring**: Tracks file system changes to detect downloads
- **Error Recovery**: Comprehensive exception handling
- **File Renaming**: Renames downloaded files to meaningful names

**Why This Matters**: This function handles the complexity of PDF downloads across different manufacturer websites, each with their own download mechanisms.

### 8. `search_and_download()` Function

**Purpose**: Orchestrates the complete search and download process for a manufacturer and component type.

**Algorithm**:
```python
def search_and_download(self, manufacturer, component_type=None, max_results=50):
    query = self.generate_query(manufacturer, component_type)
    driver = self.setup_browser(download_dir)
    
    results = []
    page = 0
    
    # Multi-page search loop
    while len(results) < max_results:
        time.sleep(self.get_random_delay())
        url = f"https://www.google.com/search?q={query}&start={page*10}"
        driver.get(url)
        
        if not self.handle_captcha(driver):
            break
        
        # Wait for search results
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#search")))
        
        # Random scrolling behavior
        for i in range(random.randint(2, 5)):
            scroll_amount = random.randint(100, 1000)
            driver.execute_script(f"window.scrollTo(0, {scroll_amount});")
            time.sleep(random.uniform(0.5, 2))
        
        # Extract links from search results
        search_div = driver.find_element(By.ID, "search")
        links = search_div.find_elements(By.TAG_NAME, "a")
        
        for link in links:
            url = link.get_attribute("href")
            title = link.text or "Untitled"
            
            # Filter by manufacturer URL patterns
            url_pattern = self.manufacturer_patterns.get(manufacturer, "/document/")
            if url_pattern in url.lower():
                results.append({'url': url, 'title': title})
                if len(results) >= max_results:
                    break
        
        page += 1
        time.sleep(random.uniform(10, 20))  # Delay between pages
    
    # Download all found datasheets
    downloaded = []
    for result in results:
        success, filename = self.download_datasheet(
            result['url'], manufacturer, result['title'], component_type
        )
        if success:
            downloaded.append(result)
        time.sleep(random.uniform(1, 2))  # Delay between downloads
    
    return downloaded
```

**Key Features**:
- **Multi-page Search**: Iterates through multiple Google search result pages
- **Random Behavior**: Implements human-like scrolling and timing
- **URL Filtering**: Only processes URLs matching manufacturer patterns
- **Sequential Downloads**: Downloads files one by one with delays
- **Progress Tracking**: Provides detailed progress feedback

**Why This Matters**: This function coordinates the entire workflow from search to download, implementing best practices for web scraping to avoid detection.

## Anti-Detection Measures

### 1. User Agent Rotation
- **5 Different User Agents**: Rotates between Chrome, Firefox, and Safari signatures
- **Random Selection**: Uses `random.choice()` for unpredictable agent selection
- **CDP Override**: Uses Chrome DevTools Protocol to override user agent

### 2. Timing Randomization
- **Request Delays**: Random delays between 5-15 seconds
- **Page Delays**: 10-20 second delays between search result pages
- **Download Delays**: 1-2 second delays between individual downloads
- **Break Periods**: 60-120 second breaks every few component types

### 3. Behavioral Mimicry
- **Random Scrolling**: Simulates human scrolling behavior
- **Random Window Sizes**: Varies browser window dimensions
- **Dynamic Content Loading**: Waits for dynamic content to load
- **Natural Navigation**: Implements realistic page navigation patterns

### 4. Technical Stealth
- **Automation Hiding**: Disables automation detection features
- **WebDriver Masking**: Hides WebDriver properties from JavaScript
- **Extension Disabling**: Disables automation-related browser extensions
- **Feature Disabling**: Disables GPU acceleration and other automation indicators

## Error Handling and Robustness

### 1. Network Error Handling
- **Timeout Management**: 30-second timeouts for page loads
- **Connection Retries**: Automatic retry logic for failed requests
- **Graceful Degradation**: Continues processing even if individual downloads fail

### 2. File System Error Handling
- **Directory Creation**: Automatic creation of missing directories
- **File Conflict Resolution**: Handles duplicate filenames gracefully
- **Permission Handling**: Manages file system permission issues

### 3. Browser Error Handling
- **CAPTCHA Detection**: Automatic detection and user intervention
- **Page Load Failures**: Timeout handling for slow-loading pages
- **Element Not Found**: Graceful handling of missing page elements

## Performance Considerations

### 1. Memory Management
- **Singleton Browser**: Reuses single WebDriver instance
- **File Monitoring**: Efficient file system change detection
- **Resource Cleanup**: Proper cleanup of browser resources

### 2. Network Optimization
- **Connection Reuse**: Reuses browser connections when possible
- **Request Batching**: Groups related operations together
- **Timeout Optimization**: Balanced timeouts for different operations

### 3. Storage Optimization
- **Duplicate Detection**: Prevents downloading existing files
- **Efficient File Naming**: Generates meaningful, unique filenames
- **Directory Structure**: Organized hierarchy for easy navigation

## Usage Examples

### Basic Usage
```python
downloader = DatasheetDownloader()
downloader.search_and_download("ti.com", "microcontroller", max_results=25)
```

### Custom Configuration
```python
downloader = DatasheetDownloader(base_download_dir="custom_datasheets")
downloader.search_and_download("analog.com", "opamp", max_results=50)
```

### Batch Processing
```python
manufacturers = ["ti.com", "analog.com", "st.com"]
component_types = ["microcontroller", "opamp", "sensor"]

for manufacturer in manufacturers:
    for component_type in component_types:
        downloader.search_and_download(manufacturer, component_type)
```

## Risks and Gotchas

### 1. Rate Limiting
- **Risk**: Manufacturers may implement rate limiting
- **Mitigation**: Random delays and break periods
- **Monitoring**: Watch for CAPTCHA challenges

### 2. Website Changes
- **Risk**: Manufacturer websites may change structure
- **Mitigation**: Flexible URL pattern matching
- **Maintenance**: Regular updates to manufacturer patterns

### 3. Legal Considerations
- **Risk**: Potential terms of service violations
- **Mitigation**: Respectful scraping practices
- **Compliance**: Follow robots.txt and rate limits

### 4. Resource Usage
- **Risk**: High memory and CPU usage
- **Mitigation**: Proper resource cleanup
- **Monitoring**: Watch system resources during execution

## Integration Points

### 1. File System Integration
- **Directory Structure**: Creates organized folder hierarchy
- **File Naming**: Generates consistent, meaningful filenames
- **Duplicate Handling**: Integrates with cleanup utilities

### 2. Web Scraping Integration
- **Search Engine Integration**: Uses Google search for discovery
- **Manufacturer Integration**: Handles multiple manufacturer websites
- **Content Extraction**: Extracts metadata from search results

### 3. Automation Integration
- **Selenium Integration**: Uses WebDriver for browser automation
- **Chrome Integration**: Leverages Chrome-specific features
- **Download Integration**: Handles automatic file downloads

This script represents a sophisticated web scraping solution that balances automation capabilities with anti-detection measures, making it suitable for large-scale datasheet collection while respecting website resources and terms of service.
