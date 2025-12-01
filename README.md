# Flash Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready web application for detecting **flash defects** in industrial parts using computer vision. Compare test images against a golden reference sample to automatically identify and quantify manufacturing defects.

![Flash Detector Screenshot](docs/screenshot.png)

## 🎯 Features

### Core Detection Capabilities
- **Automatic Image Alignment** - Uses SIFT/ORB feature matching with homography transformation to handle rotation, translation, and scale differences between images
- **Multiple Binarization Methods** - Otsu's automatic thresholding, manual threshold control, or adaptive thresholding for varying illumination
- **Morphological Operations** - Clean up binary images with open/close/dilate/erode operations
- **ROI Detection** - Automatically detect circular regions of interest
- **Flash Quantification** - Calculate precise flash percentage, pixel counts, and IoU metrics

### Web Interface
- **Real-time Parameter Tuning** - Adjust all detection parameters with instant visual feedback
- **Drag & Drop Upload** - Easy image upload for both reference and test images
- **Visual Results** - Comprehensive visualization with overlay, flash mask, and composite views
- **Responsive Design** - Works on desktop and tablet devices

### Production Ready
- **RESTful API** - Full API for integration with existing systems
- **Docker Support** - Containerized deployment with health checks
- **Batch Processing** - Process multiple test images at once
- **Configurable** - Extensive parameter configuration for different use cases

## 📋 Requirements

- Python 3.8 or higher
- OpenCV 4.8+
- Flask 2.3+

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
docker run -p 5000:5000 flash-detector
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
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## 📖 Usage Guide

### Web Interface

1. **Upload Reference Image**
   - Click on "Reference Image" panel or drag & drop a golden sample image
   - This should be a defect-free part that serves as the comparison baseline

2. **Adjust Parameters**
   - Use the left panel to tune detection parameters
   - Parameters update in real-time - watch the reference preview change
   - Key parameters to tune:
     - **Threshold Method**: Start with "otsu" for automatic thresholding
     - **Morph Operations**: Enable to clean up noise
     - **ROI Margin**: Adjust to exclude edge artifacts

3. **Upload Test Image**
   - Upload an image to inspect for flash defects
   - Results appear automatically showing flash percentage and location

4. **Interpret Results**
   - **Flash Percentage**: Primary metric - percentage of reference openings blocked by flash
   - **IoU**: Alignment quality indicator (higher = better alignment)
   - **Visual Overlay**: Yellow = good match, Red = flash defect

### API Usage

```python
import requests

# Set detection parameters
params = {
    "threshold_method": "otsu",
    "morph_enabled": True,
    "morph_kernel_size": 3,
    "min_flash_area": 100
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

## ⚙️ Configuration Parameters

### Binarization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_method` | string | "otsu" | Method: "otsu", "manual", or "adaptive" |
| `manual_threshold` | int | 128 | Threshold value for manual method (0-255) |
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

### Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_reference_threshold` | bool | true | Apply reference threshold to test image |
| `min_flash_area` | int | 100 | Minimum pixels to count as flash |
| `alignment_method` | string | "sift" | Feature matching: "sift", "orb", or "none" |
| `ransac_threshold` | float | 5.0 | RANSAC outlier rejection threshold |

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
| POST | `/api/detect` | Run detection on test image |
| POST | `/api/batch` | Batch detection on multiple images |

### Response Format

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
    "threshold_value": 128
  },
  "images": {
    "reference_binary": "<base64>",
    "test_binary": "<base64>",
    "flash_mask": "<base64>",
    "overlay": "<base64>",
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
│       └── index.html       # Web UI
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

# Run in debug mode
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for the excellent computer vision library
- Flask team for the lightweight web framework
- Solomon-3D for the industrial vision expertise

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/flash-detector/issues)
- **Email**: support@example.com

---

Made with ❤️ for industrial quality inspection
