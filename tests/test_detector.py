"""
Unit tests for Flash Detector core module.
"""

import pytest
import numpy as np
import cv2
from app.core import FlashDetector, DetectionParams, DetectionResult


class TestDetectionParams:
    """Tests for DetectionParams dataclass."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = DetectionParams()
        assert params.threshold_method == "otsu"
        assert params.morph_enabled == True
        assert params.roi_enabled == True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = DetectionParams(manual_threshold=150)
        d = params.to_dict()
        assert d["manual_threshold"] == 150
        assert "threshold_method" in d
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"threshold_method": "manual", "manual_threshold": 100}
        params = DetectionParams.from_dict(data)
        assert params.threshold_method == "manual"
        assert params.manual_threshold == 100


class TestFlashDetector:
    """Tests for FlashDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return FlashDetector()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a 200x200 grayscale image with a circle pattern
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 80, 255, -1)  # White circle
        cv2.circle(img, (100, 100), 40, 0, -1)    # Black center
        return img
    
    @pytest.fixture
    def sample_image_with_defect(self, sample_image):
        """Create a sample image with a simulated defect."""
        img = sample_image.copy()
        # Add a "flash" defect - block part of the white area
        cv2.rectangle(img, (60, 60), (80, 80), 0, -1)
        return img
    
    def test_set_reference(self, detector, sample_image):
        """Test setting reference image."""
        info = detector.set_reference(sample_image)
        
        assert detector.reference_image is not None
        assert detector.reference_binary is not None
        assert "threshold" in info
        assert "roi_center" in info
    
    def test_detect_without_reference(self, detector, sample_image):
        """Test that detection fails without reference."""
        with pytest.raises(ValueError, match="Reference image not set"):
            detector.detect(sample_image)
    
    def test_detect_with_reference(self, detector, sample_image, sample_image_with_defect):
        """Test flash detection."""
        detector.set_reference(sample_image)
        result = detector.detect(sample_image_with_defect)
        
        assert isinstance(result, DetectionResult)
        assert result.flash_percentage >= 0
        assert result.flash_mask is not None
    
    def test_detect_identical_images(self, detector, sample_image):
        """Test detection with identical images (no defects)."""
        detector.set_reference(sample_image)
        result = detector.detect(sample_image.copy())
        
        # Should have very low flash percentage
        assert result.flash_percentage < 5.0
        assert result.iou > 0.9
    
    def test_binarization_methods(self, detector, sample_image):
        """Test different binarization methods."""
        for method in ["otsu", "manual", "adaptive"]:
            params = DetectionParams(threshold_method=method)
            info = detector.set_reference(sample_image, params)
            assert detector.reference_binary is not None
    
    def test_morphological_operations(self, detector, sample_image):
        """Test morphological operations."""
        for op in ["open", "close", "dilate", "erode"]:
            params = DetectionParams(
                morph_enabled=True,
                morph_operation=op,
                morph_kernel_size=3
            )
            detector.set_reference(sample_image, params)
            assert detector.reference_binary is not None
    
    def test_alignment_disabled(self, detector, sample_image, sample_image_with_defect):
        """Test detection without alignment."""
        params = DetectionParams(alignment_method="none")
        detector.set_reference(sample_image, params)
        result = detector.detect(sample_image_with_defect, params)
        
        assert result.rotation_degrees == 0
        assert result.translation_x == 0
        assert result.translation_y == 0
    
    def test_result_to_dict(self, detector, sample_image, sample_image_with_defect):
        """Test result serialization."""
        detector.set_reference(sample_image)
        result = detector.detect(sample_image_with_defect)
        
        d = result.to_dict()
        assert "flash_percentage" in d
        assert "iou" in d
        assert isinstance(d["flash_percentage"], float)


class TestImageProcessing:
    """Tests for image processing functions."""
    
    def test_color_to_grayscale(self):
        """Test that color images are converted correctly."""
        detector = FlashDetector()
        
        # Create a color image
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :, 0] = 100  # Blue channel
        
        detector.set_reference(color_img)
        assert len(detector.reference_image.shape) == 2  # Should be grayscale
    
    def test_roi_detection(self):
        """Test ROI detection on circular pattern."""
        detector = FlashDetector()
        
        # Create image with clear circle
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 60, 0, -1)
        
        params = DetectionParams(roi_enabled=True, roi_margin=5)
        detector.set_reference(img, params)
        
        # ROI should be detected near center
        assert detector.roi_center is not None
        assert abs(detector.roi_center[0] - 100) < 20
        assert abs(detector.roi_center[1] - 100) < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
