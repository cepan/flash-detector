"""
Flash Detection Core Module

This module provides the core functionality for detecting flash defects
in industrial parts by comparing test images against a golden reference.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    """Parameters for flash detection algorithm."""
    
    # Binarization parameters
    threshold_method: str = "otsu"  # "otsu", "manual", "adaptive"
    manual_threshold: int = 128
    adaptive_block_size: int = 51
    adaptive_c: int = 5
    
    # Morphological operations
    morph_enabled: bool = True
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    morph_operation: str = "close"  # "open", "close", "dilate", "erode"
    
    # ROI (Region of Interest)
    roi_enabled: bool = True
    roi_margin: int = 10  # pixels to shrink from detected circle
    
    # Alignment
    alignment_method: str = "sift"  # "sift", "orb", "none"
    ransac_threshold: float = 5.0
    
    # Flash detection
    min_flash_area: int = 100  # minimum pixels to count as flash
    
    # Use reference threshold for test image
    use_reference_threshold: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "threshold_method": self.threshold_method,
            "manual_threshold": self.manual_threshold,
            "adaptive_block_size": self.adaptive_block_size,
            "adaptive_c": self.adaptive_c,
            "morph_enabled": self.morph_enabled,
            "morph_kernel_size": self.morph_kernel_size,
            "morph_iterations": self.morph_iterations,
            "morph_operation": self.morph_operation,
            "roi_enabled": self.roi_enabled,
            "roi_margin": self.roi_margin,
            "alignment_method": self.alignment_method,
            "ransac_threshold": self.ransac_threshold,
            "min_flash_area": self.min_flash_area,
            "use_reference_threshold": self.use_reference_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DetectionResult:
    """Results from flash detection."""
    
    # Metrics
    flash_percentage: float = 0.0
    flash_pixels: int = 0
    reference_opening_pixels: int = 0
    test_opening_pixels: int = 0
    iou: float = 0.0
    extra_pixels: int = 0
    
    # Alignment info
    rotation_degrees: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    scale: float = 1.0
    
    # Threshold used
    threshold_value: float = 0.0
    
    # Images (as numpy arrays)
    reference_binary: Optional[np.ndarray] = None
    test_binary_aligned: Optional[np.ndarray] = None
    flash_mask: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary (without images)."""
        return {
            "flash_percentage": round(self.flash_percentage, 2),
            "flash_pixels": self.flash_pixels,
            "reference_opening_pixels": self.reference_opening_pixels,
            "test_opening_pixels": self.test_opening_pixels,
            "iou": round(self.iou * 100, 2),
            "extra_pixels": self.extra_pixels,
            "rotation_degrees": round(self.rotation_degrees, 2),
            "translation_x": round(self.translation_x, 1),
            "translation_y": round(self.translation_y, 1),
            "scale": round(self.scale, 4),
            "threshold_value": round(self.threshold_value, 1),
        }


class FlashDetector:
    """
    Flash detector for comparing test images against a golden reference.
    
    Usage:
        detector = FlashDetector()
        detector.set_reference(reference_image)
        result = detector.detect(test_image, params)
    """
    
    def __init__(self):
        self.reference_image: Optional[np.ndarray] = None
        self.reference_binary: Optional[np.ndarray] = None
        self.reference_threshold: Optional[float] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.roi_center: Optional[Tuple[int, int]] = None
        self.roi_radius: Optional[int] = None
        
    def set_reference(self, image: np.ndarray, params: DetectionParams = None) -> Dict[str, Any]:
        """
        Set the reference (golden) image.
        
        Args:
            image: Reference image (grayscale or BGR)
            params: Detection parameters
            
        Returns:
            Dictionary with reference image info
        """
        if params is None:
            params = DetectionParams()
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            self.reference_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.reference_image = image.copy()
        
        # Binarize reference
        self.reference_binary, self.reference_threshold = self._binarize(
            self.reference_image, params
        )
        
        # Apply morphological operations
        if params.morph_enabled:
            self.reference_binary = self._apply_morphology(
                self.reference_binary, params
            )
        
        # Detect ROI
        if params.roi_enabled:
            self.roi_mask, self.roi_center, self.roi_radius = self._detect_roi(
                self.reference_binary, params.roi_margin
            )
        else:
            h, w = self.reference_binary.shape
            self.roi_mask = np.ones((h, w), dtype=np.uint8) * 255
            self.roi_center = (w // 2, h // 2)
            self.roi_radius = min(h, w) // 2
            
        return {
            "threshold": self.reference_threshold,
            "roi_center": self.roi_center,
            "roi_radius": self.roi_radius,
            "image_shape": self.reference_image.shape,
        }
    
    def detect(self, test_image: np.ndarray, params: DetectionParams = None) -> DetectionResult:
        """
        Detect flash in test image compared to reference.
        
        Args:
            test_image: Test image to analyze
            params: Detection parameters
            
        Returns:
            DetectionResult with metrics and images
        """
        if self.reference_image is None:
            raise ValueError("Reference image not set. Call set_reference() first.")
        
        if params is None:
            params = DetectionParams()
            
        result = DetectionResult()
        
        # Convert to grayscale if needed
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Align test image to reference
        test_aligned, alignment_info = self._align_images(
            self.reference_image, test_gray, params
        )
        
        result.rotation_degrees = alignment_info.get("rotation", 0.0)
        result.translation_x = alignment_info.get("tx", 0.0)
        result.translation_y = alignment_info.get("ty", 0.0)
        result.scale = alignment_info.get("scale", 1.0)
        
        # Binarize test image
        if params.use_reference_threshold and self.reference_threshold is not None:
            # Use reference threshold
            _, test_binary = cv2.threshold(
                test_aligned, self.reference_threshold, 255, cv2.THRESH_BINARY
            )
            result.threshold_value = self.reference_threshold
        else:
            test_binary, thresh = self._binarize(test_aligned, params)
            result.threshold_value = thresh
        
        # Apply morphological operations
        if params.morph_enabled:
            test_binary = self._apply_morphology(test_binary, params)
        
        # Calculate flash metrics
        metrics = self._calculate_metrics(
            self.reference_binary, test_binary, self.roi_mask, params
        )
        
        result.flash_percentage = metrics["flash_percentage"]
        result.flash_pixels = metrics["flash_pixels"]
        result.reference_opening_pixels = metrics["ref_openings"]
        result.test_opening_pixels = metrics["test_openings"]
        result.iou = metrics["iou"]
        result.extra_pixels = metrics["extra_pixels"]
        
        # Store images
        result.reference_binary = self.reference_binary
        result.test_binary_aligned = test_binary
        result.flash_mask = metrics["flash_mask"]
        result.overlay_image = self._create_overlay(
            self.reference_binary, test_binary, metrics["flash_mask"], self.roi_mask
        )
        
        return result
    
    def _binarize(self, image: np.ndarray, params: DetectionParams) -> Tuple[np.ndarray, float]:
        """Binarize image using specified method."""
        if params.threshold_method == "otsu":
            thresh, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif params.threshold_method == "manual":
            thresh = params.manual_threshold
            _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        elif params.threshold_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, params.adaptive_block_size, params.adaptive_c
            )
            thresh = -1  # Not applicable for adaptive
        else:
            raise ValueError(f"Unknown threshold method: {params.threshold_method}")
        
        return binary, thresh
    
    def _apply_morphology(self, binary: np.ndarray, params: DetectionParams) -> np.ndarray:
        """Apply morphological operations."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (params.morph_kernel_size, params.morph_kernel_size)
        )
        
        if params.morph_operation == "open":
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 
                                    iterations=params.morph_iterations)
        elif params.morph_operation == "close":
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                    iterations=params.morph_iterations)
        elif params.morph_operation == "dilate":
            return cv2.dilate(binary, kernel, iterations=params.morph_iterations)
        elif params.morph_operation == "erode":
            return cv2.erode(binary, kernel, iterations=params.morph_iterations)
        else:
            return binary
    
    def _detect_roi(self, binary: np.ndarray, margin: int) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """Detect circular ROI from binary image."""
        h, w = binary.shape
        
        # Find contours of the dark region (material)
        contours, _ = cv2.findContours(
            255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            center = (int(cx), int(cy))
            radius = int(radius) - margin
        else:
            center = (w // 2, h // 2)
            radius = min(h, w) // 2 - margin
        
        # Create circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = (dist_from_center <= radius).astype(np.uint8) * 255
        
        return mask, center, radius
    
    def _align_images(self, ref: np.ndarray, test: np.ndarray, 
                      params: DetectionParams) -> Tuple[np.ndarray, Dict]:
        """Align test image to reference using feature matching."""
        if params.alignment_method == "none":
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}
        
        # Select feature detector
        if params.alignment_method == "sift":
            detector = cv2.SIFT_create(nfeatures=3000)
        else:  # orb
            detector = cv2.ORB_create(nfeatures=3000)
        
        # Detect and compute
        kp1, des1 = detector.detectAndCompute(ref, None)
        kp2, des2 = detector.detectAndCompute(test, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.warning("Not enough keypoints for alignment")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}
        
        # Match features
        if params.alignment_method == "sift":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            logger.warning("Feature matching failed")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}
        
        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            logger.warning(f"Only {len(good_matches)} good matches found")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography (test -> reference)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, params.ransac_threshold)
        
        if H is None:
            logger.warning("Homography estimation failed")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}
        
        # Apply transform
        h, w = ref.shape[:2]
        aligned = cv2.warpPerspective(test, H, (w, h), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Extract parameters
        rotation = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
        scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        tx, ty = H[0, 2], H[1, 2]
        
        return aligned, {"rotation": rotation, "tx": tx, "ty": ty, "scale": scale}
    
    def _calculate_metrics(self, ref_binary: np.ndarray, test_binary: np.ndarray,
                           mask: np.ndarray, params: DetectionParams) -> Dict:
        """Calculate flash detection metrics."""
        # Apply mask
        ref_masked = np.where(mask > 0, ref_binary, 0)
        test_masked = np.where(mask > 0, test_binary, 0)
        
        # Openings (white regions)
        ref_openings = ref_masked == 255
        test_openings = test_masked == 255
        
        # Flash = reference open but test blocked
        flash_mask = (ref_openings & ~test_openings).astype(np.uint8) * 255
        
        # Extra = test open but reference blocked
        extra_mask = (~ref_openings & test_openings).astype(np.uint8) * 255
        
        # Filter small flash regions
        if params.min_flash_area > 0:
            contours, _ = cv2.findContours(flash_mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            flash_mask_filtered = np.zeros_like(flash_mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= params.min_flash_area:
                    cv2.drawContours(flash_mask_filtered, [cnt], -1, 255, -1)
            flash_mask = flash_mask_filtered
        
        # Calculate metrics
        ref_opening_pixels = np.sum(ref_openings)
        test_opening_pixels = np.sum(test_openings)
        flash_pixels = np.sum(flash_mask > 0)
        extra_pixels = np.sum(extra_mask > 0)
        
        intersection = np.sum(ref_openings & test_openings)
        union = np.sum(ref_openings | test_openings)
        iou = intersection / union if union > 0 else 0
        
        flash_percentage = (flash_pixels / ref_opening_pixels * 100) if ref_opening_pixels > 0 else 0
        
        return {
            "flash_percentage": flash_percentage,
            "flash_pixels": int(flash_pixels),
            "ref_openings": int(ref_opening_pixels),
            "test_openings": int(test_opening_pixels),
            "iou": float(iou),
            "extra_pixels": int(extra_pixels),
            "flash_mask": flash_mask,
        }
    
    def _create_overlay(self, ref_binary: np.ndarray, test_binary: np.ndarray,
                        flash_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Create visualization overlay."""
        h, w = ref_binary.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Green = reference openings, Red = test openings
        # Yellow = both (good match)
        overlay[:, :, 1] = np.where(roi_mask > 0, ref_binary, 0)  # Green
        overlay[:, :, 2] = np.where(roi_mask > 0, test_binary, 0)  # Red
        
        # Highlight flash in bright red
        overlay[flash_mask > 0] = [0, 0, 255]
        
        return overlay


def process_single_image(ref_path: str, test_path: str, 
                         params: DetectionParams = None) -> DetectionResult:
    """
    Convenience function to process a single image pair.
    
    Args:
        ref_path: Path to reference image
        test_path: Path to test image
        params: Detection parameters
        
    Returns:
        DetectionResult
    """
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference image: {ref_path}")
    if test_img is None:
        raise FileNotFoundError(f"Could not load test image: {test_path}")
    
    detector = FlashDetector()
    detector.set_reference(ref_img, params)
    return detector.detect(test_img, params)
