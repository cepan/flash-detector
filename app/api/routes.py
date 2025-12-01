"""
Flask API for Flash Detection

Provides REST endpoints for uploading images, configuring parameters,
and running flash detection analysis.
"""

import os
import cv2
import numpy as np
import base64
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..core import FlashDetector, DetectionParams

api = Blueprint('api', __name__)

# Global detector instance
detector = FlashDetector()
current_params = DetectionParams()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_base64(image: np.ndarray, format: str = '.png') -> str:
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_image_base64(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy image."""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"})


@api.route('/params', methods=['GET'])
def get_params():
    """Get current detection parameters."""
    return jsonify(current_params.to_dict())


@api.route('/params', methods=['POST'])
def set_params():
    """Update detection parameters."""
    global current_params
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    try:
        current_params = DetectionParams.from_dict(data)
        return jsonify({"status": "success", "params": current_params.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@api.route('/params/reset', methods=['POST'])
def reset_params():
    """Reset parameters to defaults."""
    global current_params
    current_params = DetectionParams()
    return jsonify({"status": "success", "params": current_params.to_dict()})


@api.route('/reference', methods=['POST'])
def upload_reference():
    """Upload reference (golden) image."""
    global detector, current_params
    
    # Check for file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            # Read image from file
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            return jsonify({"error": "Invalid file type"}), 400
    
    # Check for base64 image
    elif request.is_json and 'image' in request.get_json():
        data = request.get_json()
        try:
            image = decode_image_base64(data['image'])
        except Exception as e:
            return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400
    else:
        return jsonify({"error": "No image provided"}), 400
    
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400
    
    try:
        # Set reference image
        info = detector.set_reference(image, current_params)
        
        # Get binary preview
        binary_preview = encode_image_base64(detector.reference_binary)
        
        return jsonify({
            "status": "success",
            "info": {
                "threshold": info["threshold"],
                "roi_center": list(info["roi_center"]),
                "roi_radius": info["roi_radius"],
                "image_shape": list(info["image_shape"]),
            },
            "preview": binary_preview,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route('/reference/preview', methods=['GET'])
def get_reference_preview():
    """Get reference image preview with current parameters."""
    global detector, current_params
    
    if detector.reference_image is None:
        return jsonify({"error": "No reference image set"}), 400
    
    try:
        # Re-process with current params
        info = detector.set_reference(detector.reference_image, current_params)
        binary_preview = encode_image_base64(detector.reference_binary)
        original_preview = encode_image_base64(detector.reference_image)
        
        # Create ROI visualization
        roi_viz = cv2.cvtColor(detector.reference_image, cv2.COLOR_GRAY2BGR)
        if detector.roi_center and detector.roi_radius:
            cv2.circle(roi_viz, detector.roi_center, detector.roi_radius, (0, 255, 0), 2)
        roi_preview = encode_image_base64(roi_viz)
        
        return jsonify({
            "status": "success",
            "info": {
                "threshold": info["threshold"],
                "roi_center": list(info["roi_center"]),
                "roi_radius": info["roi_radius"],
            },
            "original": original_preview,
            "binary": binary_preview,
            "roi": roi_preview,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route('/detect', methods=['POST'])
def detect_flash():
    """Run flash detection on test image."""
    global detector, current_params
    
    if detector.reference_image is None:
        return jsonify({"error": "No reference image set. Upload reference first."}), 400
    
    # Check for file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            return jsonify({"error": "Invalid file type"}), 400
    
    # Check for base64 image
    elif request.is_json and 'image' in request.get_json():
        data = request.get_json()
        try:
            image = decode_image_base64(data['image'])
        except Exception as e:
            return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400
    else:
        return jsonify({"error": "No image provided"}), 400
    
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400
    
    try:
        # Run detection
        result = detector.detect(image, current_params)
        
        # Encode result images
        images = {
            "reference_binary": encode_image_base64(result.reference_binary),
            "test_binary": encode_image_base64(result.test_binary_aligned),
            "flash_mask": encode_image_base64(result.flash_mask),
            "overlay": encode_image_base64(result.overlay_image),
        }
        
        # Create test + flash visualization
        # Show test binary with flash defects highlighted in red
        test_flash_viz = cv2.cvtColor(result.test_binary_aligned, cv2.COLOR_GRAY2BGR)
        test_flash_viz[result.flash_mask > 0] = [0, 0, 255]  # Highlight flash in red
        images["test_flash"] = encode_image_base64(test_flash_viz)

        # Create composite visualization (for backward compatibility)
        h, w = result.reference_binary.shape
        composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Top-left: reference binary
        composite[:h, :w] = cv2.cvtColor(result.reference_binary, cv2.COLOR_GRAY2BGR)

        # Top-right: test binary (aligned)
        composite[:h, w:] = cv2.cvtColor(result.test_binary_aligned, cv2.COLOR_GRAY2BGR)

        # Bottom-left: test + flash
        composite[h:, :w] = test_flash_viz

        # Bottom-right: overlay (green=ref, red=test, yellow=match, bright red=flash)
        composite[h:, w:] = result.overlay_image

        # Resize for web display
        scale = min(1200 / composite.shape[1], 900 / composite.shape[0])
        if scale < 1:
            composite = cv2.resize(composite, None, fx=scale, fy=scale)

        images["composite"] = encode_image_base64(composite)
        
        return jsonify({
            "status": "success",
            "result": result.to_dict(),
            "images": images,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api.route('/batch', methods=['POST'])
def batch_detect():
    """Run batch detection on multiple test images."""
    global detector, current_params
    
    if detector.reference_image is None:
        return jsonify({"error": "No reference image set"}), 400
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    result = detector.detect(image, current_params)
                    results.append({
                        "filename": secure_filename(file.filename),
                        "result": result.to_dict(),
                        "flash_mask": encode_image_base64(result.flash_mask),
                    })
            except Exception as e:
                results.append({
                    "filename": secure_filename(file.filename),
                    "error": str(e),
                })
    
    return jsonify({
        "status": "success",
        "count": len(results),
        "results": results,
    })
