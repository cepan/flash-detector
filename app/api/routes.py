"""
Flask API for Flash Detection

Provides REST endpoints for uploading images, configuring parameters,
and running flash detection analysis.
"""

import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..core import FlashDetector, DetectionParams

api = Blueprint('api', __name__)

# Global detector instance
detector = FlashDetector()
current_params = DetectionParams()
original_reference_image = None  # Store original unrotated reference for rotation
saved_rotation_angle = 0  # Store the applied rotation angle

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Saved data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVED_DATA_DIR = os.path.join(PROJECT_ROOT, 'saved_data')
SAVED_REFERENCE_PATH = os.path.join(SAVED_DATA_DIR, 'reference_original.png')  # Original unrotated
SAVED_REFERENCE_ROTATED_PATH = os.path.join(SAVED_DATA_DIR, 'reference_rotated.png')  # After rotation
SAVED_CONFIG_PATH = os.path.join(SAVED_DATA_DIR, 'reference_config.json')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_saved_reference_on_startup():
    """Load saved reference on application startup if it exists."""
    global original_reference_image, saved_rotation_angle, detector, current_params

    if not os.path.exists(SAVED_REFERENCE_PATH) or not os.path.exists(SAVED_CONFIG_PATH):
        print("No saved reference found. Starting fresh.")
        return

    try:
        # Load configuration
        with open(SAVED_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        # Load original image
        original_image = cv2.imread(SAVED_REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print("Failed to load saved reference image.")
            return

        original_reference_image = original_image
        saved_rotation_angle = config.get("rotation_angle", 0)

        # Try to load the pre-rotated image first (faster!)
        if os.path.exists(SAVED_REFERENCE_ROTATED_PATH):
            rotated = cv2.imread(SAVED_REFERENCE_ROTATED_PATH, cv2.IMREAD_GRAYSCALE)
            if rotated is not None:
                print(f"✓ Loaded pre-rotated reference image")
            else:
                # Fallback: calculate rotation
                rotated = original_image
        else:
            # No rotated image saved, use original
            rotated = original_image

        # Set reference in detector
        detector.set_reference(rotated, current_params)

        print(f"✓ Auto-loaded saved reference (rotation: {saved_rotation_angle}°)")
        print(f"  Timestamp: {config.get('upload_timestamp', 'unknown')}")
        print(f"  Image shape: {config.get('image_shape', 'unknown')}")

    except Exception as e:
        print(f"Error loading saved reference: {e}")


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
    global detector, current_params, original_reference_image

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
        # Store original unrotated reference image for rotation operations
        original_reference_image = image.copy()
        saved_rotation_angle = 0  # Reset rotation angle when new image uploaded

        # Set reference image
        info = detector.set_reference(image, current_params)

        # Get binary preview
        binary_preview = encode_image_base64(detector.reference_binary)

        # Auto-save reference when uploaded
        try:
            os.makedirs(SAVED_DATA_DIR, exist_ok=True)
            cv2.imwrite(SAVED_REFERENCE_PATH, original_reference_image)

            # Save rotated version (same as original on first upload)
            cv2.imwrite(SAVED_REFERENCE_ROTATED_PATH, original_reference_image)

            config = {
                "rotation_angle": 0,
                "upload_timestamp": datetime.now().isoformat(),
                "image_shape": list(original_reference_image.shape),
                "rotated_image_shape": list(original_reference_image.shape),
                "threshold": float(detector.reference_threshold) if detector.reference_threshold else None,
                "params": current_params.to_dict()
            }

            with open(SAVED_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            # Don't fail the upload if save fails
            pass

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

        # Add labeled overlay if hole analysis was performed
        if result.labeled_overlay is not None:
            images["labeled_overlay"] = encode_image_base64(result.labeled_overlay)

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


@api.route('/reference/rotate', methods=['POST'])
def rotate_reference():
    """Rotate the reference image by specified angle."""
    global detector, current_params, original_reference_image, saved_rotation_angle

    if original_reference_image is None:
        return jsonify({"error": "No reference image set"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    angle = data.get('angle', 0)
    saved_rotation_angle = angle  # Store the rotation angle

    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"=== ROTATION REQUEST ===")
        logger.info(f"Requested angle: {angle} degrees")
        logger.info(f"Original image exists: {original_reference_image is not None}")
        logger.info(f"Original image shape: {original_reference_image.shape if original_reference_image is not None else 'None'}")

        # Save original before rotation for comparison
        orig_debug_path = os.path.join("/tmp/flash_detector_debug", f"original_before_rotation_{angle}.png")
        os.makedirs("/tmp/flash_detector_debug", exist_ok=True)
        cv2.imwrite(orig_debug_path, original_reference_image)
        logger.info(f"Saved original (before rotation) to: {orig_debug_path}")

        # Always rotate from the ORIGINAL unrotated reference image
        h, w = original_reference_image.shape[:2]
        center = (w // 2, h // 2)

        # Create rotation matrix
        # NOTE: OpenCV rotation is COUNTER-CLOCKWISE for positive angles
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        logger.info(f"Rotation matrix center: {center}")
        logger.info(f"Original image dimensions: {w}x{h}")

        # Calculate expanded canvas to fit entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to center the image in expanded canvas
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        logger.info(f"Expanded canvas for rotation: {new_w}x{new_h}")

        # Rotate with expanded canvas (no cropping)
        rotated = cv2.warpAffine(
            original_reference_image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  # White background
        )

        logger.info(f"Rotated image shape: {rotated.shape}")
        logger.info(f"Rotated image min/max values: {rotated.min()}/{rotated.max()}")

        # Re-process with rotated image (this updates detector.reference_image)
        info = detector.set_reference(rotated, current_params)

        logger.info(f"Detector updated with rotated image")
        logger.info(f"New threshold: {info['threshold']}")

        # Save rotated images to disk for debugging
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = "/tmp/flash_detector_debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Save the rotated grayscale image
        rotated_path = os.path.join(debug_dir, f"rotated_grayscale_{timestamp}_angle{angle}.png")
        cv2.imwrite(rotated_path, rotated)

        # Save the rotated binary image
        binary_path = os.path.join(debug_dir, f"rotated_binary_{timestamp}_angle{angle}.png")
        cv2.imwrite(binary_path, detector.reference_binary)

        logger.info(f"Saved rotated grayscale to: {rotated_path}")
        logger.info(f"Saved rotated binary to: {binary_path}")
        logger.info(f"=== ROTATION COMPLETE ===")

        # Get previews
        binary_preview = encode_image_base64(detector.reference_binary)
        original_preview = encode_image_base64(rotated)

        # Save what we're sending to frontend for debugging
        sent_binary_path = os.path.join(debug_dir, f"sent_to_frontend_binary_{timestamp}_angle{angle}.png")
        cv2.imwrite(sent_binary_path, detector.reference_binary)
        logger.info(f"Saved binary being sent to frontend: {sent_binary_path}")

        # Create ROI visualization
        roi_viz = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        if detector.roi_center and detector.roi_radius:
            cv2.circle(roi_viz, detector.roi_center, detector.roi_radius, (0, 255, 0), 2)
        roi_preview = encode_image_base64(roi_viz)

        # Auto-save reference, rotated image, and config
        try:
            os.makedirs(SAVED_DATA_DIR, exist_ok=True)

            # Save original (unrotated) image
            cv2.imwrite(SAVED_REFERENCE_PATH, original_reference_image)

            # Save rotated image for faster loading
            cv2.imwrite(SAVED_REFERENCE_ROTATED_PATH, rotated)

            config = {
                "rotation_angle": saved_rotation_angle,
                "upload_timestamp": datetime.now().isoformat(),
                "image_shape": list(original_reference_image.shape),
                "rotated_image_shape": list(rotated.shape),
                "threshold": float(detector.reference_threshold) if detector.reference_threshold else None,
                "params": current_params.to_dict()
            }

            with open(SAVED_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Auto-saved reference with rotation angle: {saved_rotation_angle}")
            logger.info(f"Saved rotated image: {SAVED_REFERENCE_ROTATED_PATH}")
        except Exception as e:
            logger.warning(f"Failed to auto-save reference: {str(e)}")

        return jsonify({
            "status": "success",
            "angle": angle,
            "info": {
                "threshold": info["threshold"],
                "roi_center": list(info["roi_center"]),
                "roi_radius": info["roi_radius"],
            },
            "original": original_preview,
            "binary": binary_preview,
            "roi": roi_preview,
            "saved": True  # Indicate that it was auto-saved
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


@api.route('/reference/save', methods=['POST'])
def save_reference():
    """Save current reference image and configuration to disk."""
    global original_reference_image, saved_rotation_angle, detector, current_params

    if original_reference_image is None:
        return jsonify({"error": "No reference image to save"}), 400

    try:
        # Ensure saved_data directory exists
        os.makedirs(SAVED_DATA_DIR, exist_ok=True)

        # Save the original (unrotated) reference image
        cv2.imwrite(SAVED_REFERENCE_PATH, original_reference_image)

        # Save configuration
        config = {
            "rotation_angle": saved_rotation_angle,
            "upload_timestamp": datetime.now().isoformat(),
            "image_shape": list(original_reference_image.shape),
            "threshold": float(detector.reference_threshold) if detector.reference_threshold else None,
            "params": current_params.to_dict()
        }

        with open(SAVED_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({
            "status": "success",
            "message": "Reference saved successfully",
            "config": config
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route('/reference/load', methods=['POST'])
def load_reference():
    """Load saved reference image and configuration from disk."""
    global original_reference_image, saved_rotation_angle, detector, current_params

    if not os.path.exists(SAVED_REFERENCE_PATH) or not os.path.exists(SAVED_CONFIG_PATH):
        return jsonify({"error": "No saved reference found"}), 404

    try:
        # Load configuration
        with open(SAVED_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        # Load original image
        original_image = cv2.imread(SAVED_REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return jsonify({"error": "Failed to load saved image"}), 500

        original_reference_image = original_image
        saved_rotation_angle = config.get("rotation_angle", 0)

        # Try to load the pre-rotated image first (faster!)
        if os.path.exists(SAVED_REFERENCE_ROTATED_PATH):
            rotated = cv2.imread(SAVED_REFERENCE_ROTATED_PATH, cv2.IMREAD_GRAYSCALE)
            if rotated is None:
                # Fallback to original
                rotated = original_image
        else:
            # No rotated image saved, use original
            rotated = original_image

        # Set reference in detector
        info = detector.set_reference(rotated, current_params)

        # Get previews
        binary_preview = encode_image_base64(detector.reference_binary)
        original_preview = encode_image_base64(rotated)

        return jsonify({
            "status": "success",
            "config": config,
            "info": {
                "threshold": info["threshold"],
                "roi_center": list(info["roi_center"]),
                "roi_radius": info["roi_radius"],
            },
            "original": original_preview,
            "binary": binary_preview,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@api.route('/reference/status', methods=['GET'])
def reference_status():
    """Check if a saved reference exists."""
    exists = os.path.exists(SAVED_REFERENCE_PATH) and os.path.exists(SAVED_CONFIG_PATH)

    config = None
    if exists:
        try:
            with open(SAVED_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except:
            pass

    return jsonify({
        "has_saved_reference": exists,
        "config": config
    })
