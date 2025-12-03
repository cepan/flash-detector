# Flash Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)](https://github.com/yourusername/flash-detector)

A production-ready web application for detecting **flash defects** in industrial parts using advanced computer vision. Compare test images against a golden reference sample to automatically identify, quantify, and analyze manufacturing defects with per-hole precision.

![Flash Detector Screenshot](docs/screenshot.png)

## 🎯 Features

### Core Detection Capabilities
- **Automatic Image Alignment** - Uses SIFT/ORB feature matching with homography transformation to handle rotation, translation, and scale differences between images
- **Manual Reference Rotation** - Grid-based rotation tool with adjustable overlay (2x2 to 20x20) for precise golden sample alignment to I-markers or reference features
- **Multiple Binarization Methods** - Otsu's automatic thresholding, manual threshold control, or adaptive thresholding for varying illumination
- **Morphological Operations** - Clean up binary images with open/close/dilate/erode operations
- **ROI Detection** - Automatically detect circular regions of interest
- **Flash Quantification** - Calculate precise flash percentage, pixel counts, and IoU metrics
- **Per-Hole Flash Analysis** ⭐ NEW - Individual hole detection, labeling, and flash percentage calculation with interactive table and CSV export

### Advanced Visualization
- **Interactive Hole Labels** ⭐ NEW - Click hole labels on overlay image to scroll to table row; click table rows to highlight holes
- **Eye-Friendly Color Palette** ⭐ NEW - Softer, muted colors (70% reduced intensity) for extended viewing sessions without eye strain
- **Labeled Overlay** - Toggle between standard overlay and labeled overlay showing hole IDs and status
- **Real-time Parameter Tuning** - Adjust all detection parameters with instant visual feedback
- **Comprehensive Results** - Visual overlay, flash mask, per-hole analysis table, and composite views

### Web Interface
- **Workflow-Oriented Design** ⭐ NEW - Parameters organized in logical processing order (Binarization → Morphology → ROI → Alignment → Detection)
- **Drag & Drop Upload** - Easy image upload for both reference and test images
- **Persistent Golden Image** ⭐ NEW - Auto-save and auto-load reference images with rotation settings
- **Sortable Data Tables** - Click column headers to sort hole analysis results
- **CSV Export** - Download per-hole flash analysis data for external reporting
- **Responsive Design** - Works on desktop and tablet devices

### Production Ready
- **RESTful API** - Full API for integration with existing systems
- **Docker Support** - Containerized deployment with health checks
- **Batch Processing** - Process multiple test images at once
- **Auto-Save/Load** ⭐ NEW - Reference images automatically persist across sessions
- **Configurable** - Extensive parameter configuration for different use cases

## 📋 Requirements

- Python 3.8 or higher
- OpenCV 4.8+
- Flask 2.3+
- NumPy 1.24+

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flash-detector.git
cd flash-detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

Open your browser to `http://localhost:5000`

### Option 2: Docker

```bash
# Build the image
docker build -t flash-detector .

# Run the container
docker run -p 5000:5000 -v $(pwd)/saved_data:/app/saved_data flash-detector
```

### Option 3: Docker Compose

```yaml
version: '3.8'
services:
  flash-detector:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./saved_data:/app/saved_data  # Persist golden images
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## 📖 Usage Guide

### Web Interface Workflow

#### 1. **Upload Reference Image**
   - Click on "Reference Image" panel or drag & drop a golden sample image
   - This should be a defect-free part that serves as the comparison baseline
   - The image is automatically saved to `saved_data/` for future sessions

#### 2. **Rotate Reference Image** (if needed)
   - Use the rotation slider (-180° to +180°) to orient the golden sample
   - **Grid Overlay**: Adjust grid size (2-20) to align I-markers or reference features
   - **Fine-Tuning**: Use -1° and +1° buttons for precise adjustments
   - **Preview**: Live preview shows rotation with grid overlay
   - **Apply**: Click "Apply" to save the rotation permanently
   - Rotation settings persist across sessions

#### 3. **Adjust Parameters**
   - Parameters are organized in logical workflow order:
     - **1. Binarization**: Convert to binary (threshold method, threshold value)
     - **2. Morphology**: Clean up binary (operations, kernel size, iterations)
     - **3. Region of Interest**: Define ROI (auto-detect, margin)
     - **4. Alignment**: Align test to reference (method, RANSAC threshold)
     - **5. Flash Detection**: Detect defects (min flash area, per-hole analysis)
   - Parameters update in real-time - watch the reference preview change
   - Key parameters to tune:
     - **Threshold Method**: Start with "manual" (119 default) or "otsu" for automatic
     - **Morph Operations**: Enable "close" to clean up noise
     - **ROI Margin**: Adjust to exclude edge artifacts
     - **Alignment Method**: Use "orb" for faster processing, "sift" for accuracy

#### 4. **Enable Per-Hole Analysis** (optional)
   - Check "Analyze Individual Holes" in Flash Detection section
   - Adjust "Min Hole Area" to filter out small artifacts (default: 50 pixels)
   - This enables individual hole labeling and detailed analysis

#### 5. **Upload Test Image**
   - Upload an image to inspect for flash defects
   - Results appear automatically showing flash percentage and location
   - Test image automatically aligns to the (rotated) golden sample

#### 6. **Interpret Results**
   - **Flash Percentage**: Primary metric - percentage of reference openings blocked by flash
   - **IoU**: Alignment quality indicator (higher = better, >70% recommended)
   - **Visual Overlay**:
     - Softer yellow: Material match (both have material)
     - Black: Hole match (both have openings)
     - Soft red: Flash defect (reference open, test blocked)
   - **Per-Hole Analysis** (if enabled):
     - **Hole Labels**: Click labels on image to scroll to that row in table
     - **Table**: Click rows to highlight holes, sort by any column
     - **Status Colors**: Green (Good <5%), Orange (Minor Flash 5-25%), Red (Flash Defect ≥25%)
     - **CSV Export**: Download detailed hole data for reporting

### API Usage

#### Basic Detection

```python
import requests

# Set detection parameters
params = {
    "threshold_method": "manual",
    "manual_threshold": 119,
    "morph_enabled": True,
    "morph_kernel_size": 3,
    "min_flash_area": 100,
    "alignment_method": "orb",
    "ransac_threshold": 5.5
}
requests.post("http://localhost:5000/api/params", json=params)

# Upload reference image
with open("golden_sample.bmp", "rb") as f:
    requests.post("http://localhost:5000/api/reference", files={"file": f})

# Detect flash in test image
with open("test_part.bmp", "rb") as f:
    response = requests.post("http://localhost:5000/api/detect", files={"file": f})
    result = response.json()
    print(f"Flash: {result['result']['flash_percentage']:.2f}%")
```

#### Rotation and Persistence

```python
# Rotate reference image
rotation_data = {"angle": -90}
response = requests.post("http://localhost:5000/api/reference/rotate", json=rotation_data)

# Check if saved reference exists
response = requests.get("http://localhost:5000/api/reference/status")
status = response.json()
if status["has_saved_reference"]:
    print(f"Saved reference found: rotation = {status['config']['rotation_angle']}°")

# Load saved reference
response = requests.post("http://localhost:5000/api/reference/load")
```

#### Per-Hole Analysis

```python
# Enable per-hole analysis
params = {
    "analyze_individual_holes": True,
    "min_hole_area": 50
}
requests.post("http://localhost:5000/api/params", json=params)

# Run detection
with open("test_part.bmp", "rb") as f:
    response = requests.post("http://localhost:5000/api/detect", files={"file": f})
    result = response.json()

    # Access hole details
    for hole in result['result']['hole_details']:
        print(f"Hole {hole['hole_id']}: {hole['flash_percentage']:.2f}% flash - {hole['status']}")
```

## ⚙️ Configuration Parameters

### Binarization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_method` | string | "manual" | Method: "otsu", "manual", or "adaptive" |
| `manual_threshold` | int | 119 | Threshold value for manual method (0-255) |
| `adaptive_block_size` | int | 51 | Block size for adaptive threshold (odd number) |
| `adaptive_c` | int | 5 | Constant subtracted from mean in adaptive |

### Morphological Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `morph_enabled` | bool | true | Enable morphological operations |
| `morph_operation` | string | "close" | Operation: "open", "close", "dilate", "erode" |
| `morph_kernel_size` | int | 3 | Kernel size (odd number, 1-15) |
| `morph_iterations` | int | 1 | Number of iterations |

### ROI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roi_enabled` | bool | true | Auto-detect circular ROI |
| `roi_margin` | int | 10 | Pixels to shrink from detected edge |

### Alignment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alignment_method` | string | "orb" | Feature matching: "sift", "orb", "ecc", or "none" |
| `ransac_threshold` | float | 5.5 | RANSAC outlier rejection threshold |

### Flash Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_reference_threshold` | bool | true | Apply reference threshold to test image |
| `min_flash_area` | int | 100 | Minimum pixels to count as flash defect |
| `analyze_individual_holes` | bool | false | Enable per-hole flash detection and labeling |
| `min_hole_area` | int | 50 | Minimum pixels to count as a hole |

### Reference Rotation Parameters

| Parameter | Description |
|-----------|-------------|
| Rotation Angle | -180° to +180° via slider or fine-tune buttons (±1°) |
| Grid Size | 2x2 to 20x20 overlay for alignment assistance |
| Auto-Save | Rotation settings automatically persist to disk |

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/params` | Get current parameters |
| POST | `/api/params` | Update parameters |
| POST | `/api/params/reset` | Reset to defaults |
| POST | `/api/reference` | Upload reference image |
| GET | `/api/reference/preview` | Get reference preview |
| POST | `/api/reference/rotate` | Rotate reference by angle |
| POST | `/api/reference/save` | Manually save reference |
| POST | `/api/reference/load` | Load saved reference |
| GET | `/api/reference/status` | Check saved reference status |
| POST | `/api/detect` | Run detection on test image |
| POST | `/api/batch` | Batch detection on multiple images |

### Response Format

#### Standard Detection Response

```json
{
  "status": "success",
  "result": {
    "flash_percentage": 12.34,
    "flash_pixels": 12345,
    "reference_opening_pixels": 100000,
    "test_opening_pixels": 87655,
    "iou": 85.5,
    "extra_pixels": 100,
    "rotation_degrees": -1.5,
    "translation_x": 10.2,
    "translation_y": -5.3,
    "scale": 0.998,
    "threshold_value": 119
  },
  "images": {
    "reference_binary": "<base64>",
    "test_binary": "<base64>",
    "flash_mask": "<base64>",
    "overlay": "<base64>",
    "test_flash": "<base64>",
    "composite": "<base64>"
  }
}
```

#### Detection Response with Per-Hole Analysis

```json
{
  "status": "success",
  "result": {
    "flash_percentage": 12.34,
    "flash_pixels": 12345,
    "reference_opening_pixels": 100000,
    "test_opening_pixels": 87655,
    "iou": 85.5,
    "hole_details": [
      {
        "hole_id": 1,
        "center_x": 450,
        "center_y": 380,
        "area": 1234,
        "flash_pixels": 156,
        "flash_percentage": 12.63,
        "status": "Minor Flash"
      },
      {
        "hole_id": 2,
        "center_x": 680,
        "center_y": 390,
        "area": 1180,
        "flash_pixels": 12,
        "flash_percentage": 1.02,
        "status": "Good"
      }
    ]
  },
  "images": {
    "reference_binary": "<base64>",
    "test_binary": "<base64>",
    "flash_mask": "<base64>",
    "overlay": "<base64>",
    "labeled_overlay": "<base64>",
    "test_flash": "<base64>",
    "composite": "<base64>"
  }
}
```

## 🏗️ Project Structure

```
flash-detector/
├── app/
│   ├── __init__.py          # Flask application factory
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── detector.py      # Core detection algorithm
│   └── static/
│       └── index.html       # React-based Web UI
├── saved_data/              # Saved reference images (gitignored)
│   ├── reference_original.png   # Original unrotated image
│   ├── reference_rotated.png    # Pre-rotated for fast loading
│   ├── reference_config.json    # Rotation angle and metadata
│   └── .gitignore
├── tests/
│   ├── __init__.py
│   ├── test_detector.py     # Unit tests for detector
│   └── test_api.py          # API integration tests
├── uploads/                  # Uploaded images (gitignored)
│   ├── reference/
│   └── test/
├── outputs/                  # Output files (gitignored)
├── docs/                     # Documentation
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py                    # Application entry point
└── README.md
```

## 🔍 Key Features Deep Dive

### Manual Rotation Tool

The rotation tool helps align golden samples to reference features (I-markers, mounting holes, etc.):

- **Adjustable Grid Overlay**: Choose from 2x2 to 20x20 grid (default: 8x8)
- **Live Preview**: See rotation applied in real-time before committing
- **Fine-Tuning**: Use -1°/+1° buttons for precise micro-adjustments
- **Persistent**: Rotation angle and rotated image saved automatically
- **Fast Loading**: Pre-rotated image loads instantly on next session

**Use Cases:**
- Align I-markers to grid lines for consistent orientation
- Correct camera mounting angle offsets
- Ensure test parts align properly regardless of fixture positioning

### Per-Hole Flash Analysis

Individual hole detection provides detailed defect analysis:

**How It Works:**
1. Contour detection identifies each hole in the reference image
2. For each hole, flash percentage is calculated independently
3. Holes are classified: Good (<5%), Minor Flash (5-25%), Flash Defect (≥25%)
4. Each hole is labeled with ID and color-coded on the overlay

**Interactive Features:**
- **Click Labels**: Click hole number on image → table scrolls to that row
- **Click Rows**: Click table row → row highlights in purple
- **Sortable Table**: Sort by hole ID, position, area, flash %, or status
- **CSV Export**: Download all hole data for quality reports

**Visual Indicators:**
- **Green** circle/label: Good hole (<5% flash)
- **Orange** circle/label: Minor flash (5-25%)
- **Red** circle/label: Flash defect (≥25%)

### Persistent Storage System

Golden images and settings automatically persist:

**What Gets Saved:**
- Original unrotated reference image
- Pre-rotated reference image (for instant loading)
- Rotation angle
- Detection parameters
- Upload timestamp
- Image dimensions

**Auto-Save Triggers:**
- Upload new reference → saves immediately
- Apply rotation → saves rotated version

**Auto-Load Behavior:**
- On startup: Checks for saved reference
- If found: Loads pre-rotated image (instant, no calculation)
- Shows green indicator: "✓ Using saved golden image"

**Benefits:**
- Never re-upload golden samples
- Consistent results across sessions
- Fast startup (pre-rotated image cached)
- Team members can share saved_data/ folder

### Eye-Friendly Color Scheme

The interface uses a carefully designed color palette to reduce eye strain:

**Overlay Colors** (70% reduced intensity):
- Soft yellow: Material match
- Black: Hole match
- Soft salmon: Flash defect

**Hole Label Colors:**
- Soft teal/green: Good holes
- Soft orange/peach: Minor flash
- Soft red/salmon: Flash defects

All colors maintain excellent contrast while being gentler on eyes during extended inspection sessions.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_detector.py -v
```

## 🔧 Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run in debug mode (auto-reload on code changes)
python run.py --debug

# Format code
black app/ tests/

# Lint
flake8 app/ tests/
```

## 📊 Performance Tips

1. **Image Size**: For faster processing, resize large images to 1920x1200 or smaller
2. **Feature Matching**: Use "orb" instead of "sift" for faster (but less accurate) alignment
3. **Batch Processing**: Use the `/api/batch` endpoint for multiple images
4. **Caching**: Reference image processing is cached - only re-processed when parameters change
5. **Saved References** ⭐: Golden images auto-load on startup - no re-upload needed
6. **Pre-Rotated Loading** ⭐: Rotated images cached for instant loading (no rotation calculation)
7. **Per-Hole Analysis**: Enable only when detailed analysis needed (adds ~100-300ms processing time)

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build production image
docker build -t flash-detector:2.0.0 .

# Run with volume mounts
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/saved_data:/app/saved_data \
  -v $(pwd)/uploads:/app/uploads \
  --name flash-detector \
  --restart unless-stopped \
  flash-detector:2.0.0
```

### Environment Variables

```bash
# Set environment variables
docker run -d \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -e MAX_CONTENT_LENGTH=52428800 \
  flash-detector:2.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Changelog

### Version 2.0.0 (2025-12-02)
- ⭐ Added per-hole flash detection and analysis
- ⭐ Added manual rotation tool with grid overlay
- ⭐ Added persistent golden image storage
- ⭐ Added interactive label/table synchronization
- ⭐ Implemented eye-friendly color palette
- ⭐ Reorganized UI in logical workflow order
- Enhanced parameter panel with better typography
- Added CSV export for hole data
- Improved text rendering with proper background wrapping

### Version 1.0.0 (2024)
- Initial release with basic flash detection
- SIFT/ORB alignment
- Multiple binarization methods
- ROI detection
- REST API

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for the excellent computer vision library
- Flask team for the lightweight web framework
- React team for the UI framework
- Solomon-3D for the industrial vision expertise

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/flash-detector/issues)
- **Documentation**: See `/docs` folder for detailed guides
- **Email**: support@example.com

---

Made with ❤️ for industrial quality inspection
