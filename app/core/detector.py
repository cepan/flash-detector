"""
Flash Detection Core Module

This module provides the core functionality for detecting flash defects
in industrial parts by comparing test images against a golden reference.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    """Parameters for flash detection algorithm."""
    
    # Binarization parameters
    threshold_method: str = "manual"  # "otsu", "manual", "adaptive"
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
    alignment_method: str = "orb"  # "sift", "orb", "ecc", "none"
    ransac_threshold: float = 5.5
    max_iterations: int = 1000  # For ECC alignment
    
    # Flash detection
    min_flash_area: int = 100  # minimum pixels to count as flash
    invert_binary: bool = False  # Invert binary after thresholding (if openings are dark)

    # Use reference threshold for test image
    use_reference_threshold: bool = True

    # Per-hole analysis
    analyze_individual_holes: bool = True  # Enable per-hole flash detection
    min_hole_area: int = 50  # Minimum area (pixels) to count as a hole
    
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
            "analyze_individual_holes": self.analyze_individual_holes,
            "min_hole_area": self.min_hole_area,
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

    # Per-hole details (optional)
    hole_details: Optional[List[Dict[str, Any]]] = None
    labeled_overlay: Optional[np.ndarray] = None

    # Images (as numpy arrays)
    reference_binary: Optional[np.ndarray] = None
    test_binary_aligned: Optional[np.ndarray] = None
    flash_mask: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary (without images)."""
        result = {
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

        # Include hole details if available
        if self.hole_details is not None:
            result["hole_details"] = self.hole_details

        return result


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

        # Per-hole analysis (if enabled)
        if params.analyze_individual_holes:
            result.hole_details = self._detect_individual_holes(
                self.reference_binary, test_binary, self.roi_mask, params
            )
            result.labeled_overlay = self._create_labeled_overlay(
                self.reference_binary, test_binary, metrics["flash_mask"],
                self.roi_mask, result.hole_details
            )

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

    def _align_ecc(self, ref: np.ndarray, test: np.ndarray,
                   params: DetectionParams) -> Tuple[np.ndarray, Dict]:
        """Align images using ECC (Enhanced Correlation Coefficient).

        Better for repetitive patterns like circular grids.
        """
        h, w = ref.shape[:2]

        # Define motion model - using Euclidean (rotation + translation)
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   params.max_iterations, 1e-6)

        try:
            # Run ECC alignment
            (cc, warp_matrix) = cv2.findTransformECC(ref, test, warp_matrix,
                                                     warp_mode, criteria,
                                                     inputMask=None, gaussFiltSize=5)

            logger.info(f"ECC alignment converged with correlation: {cc:.4f}")

            # Apply transformation
            aligned = cv2.warpAffine(test, warp_matrix, (w, h),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)

            # Extract rotation and translation
            rotation = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]) * 180 / np.pi
            tx = warp_matrix[0, 2]
            ty = warp_matrix[1, 2]
            scale = np.sqrt(warp_matrix[0, 0]**2 + warp_matrix[1, 0]**2)

            return aligned, {"rotation": rotation, "tx": tx, "ty": ty, "scale": scale}

        except cv2.error as e:
            logger.warning(f"ECC alignment failed: {str(e)}")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}

    def _align_images(self, ref: np.ndarray, test: np.ndarray,
                      params: DetectionParams) -> Tuple[np.ndarray, Dict]:
        """Align test image to reference using feature matching."""
        if params.alignment_method == "none":
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}

        # ECC (Enhanced Correlation Coefficient) - better for repetitive patterns
        if params.alignment_method == "ecc":
            return self._align_ecc(ref, test, params)

        # Select feature detector
        if params.alignment_method == "sift":
            detector = cv2.SIFT_create(nfeatures=3000)
        else:  # orb
            detector = cv2.ORB_create(nfeatures=3000)
        
        # Detect and compute
        kp1, des1 = detector.detectAndCompute(ref, None)
        kp2, des2 = detector.detectAndCompute(test, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.warning(f"Not enough keypoints for alignment: ref={len(kp1) if kp1 else 0}, test={len(kp2) if kp2 else 0}")
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
            logger.warning(f"Only {len(good_matches)} good matches found (need at least 10 for homography)")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}

        logger.info(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography (test -> reference)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, params.ransac_threshold)

        if H is None:
            logger.warning("Homography estimation failed")
            return test, {"rotation": 0, "tx": 0, "ty": 0, "scale": 1}

        # Check how many inliers we got
        inliers = np.sum(mask) if mask is not None else 0
        logger.info(f"Homography inliers: {inliers}/{len(good_matches)} ({inliers*100/len(good_matches):.1f}%)")
        
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
        """Calculate flash detection metrics.

        Flash defect logic:
        - Binary images: WHITE (255) = openings/holes, BLACK (0) = material
        - Flash = areas where reference is OPEN (white) but test is BLOCKED (black)
        - This means flash filled the opening that should exist
        """
        # Apply mask
        ref_masked = np.where(mask > 0, ref_binary, 0)
        test_masked = np.where(mask > 0, test_binary, 0)

        # Openings (white regions = 255)
        ref_openings = ref_masked == 255
        test_openings = test_masked == 255

        # Flash defect = reference is open (white) but test is blocked (black/not white)
        # This means the opening got filled with material (flash)
        flash_mask = (ref_openings & ~test_openings).astype(np.uint8) * 255

        # Extra openings = test is open but reference is blocked
        # (material was removed, or test has extra holes - usually not flash)
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
    
    def _detect_individual_holes(self, ref_binary: np.ndarray, test_binary: np.ndarray,
                                  mask: np.ndarray, params: DetectionParams) -> List[Dict[str, Any]]:
        """Detect and analyze individual holes.

        Args:
            ref_binary: Reference binary image (WHITE=openings)
            test_binary: Test binary image (WHITE=openings)
            mask: ROI mask
            params: Detection parameters

        Returns:
            List of dictionaries containing hole details
        """
        # Apply mask to reference
        ref_masked = np.where(mask > 0, ref_binary, 0)

        # Find contours of holes (white regions = openings)
        contours, _ = cv2.findContours(ref_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hole_details = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip small holes
            if area < params.min_hole_area:
                continue

            # Get bounding info
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Create mask for this specific hole
            hole_mask = np.zeros_like(ref_binary)
            cv2.drawContours(hole_mask, [contour], -1, 255, -1)

            # Count pixels for this hole in both images
            ref_opening_pixels = np.sum(hole_mask == 255)
            test_opening_pixels = np.sum((hole_mask == 255) & (test_binary == 255))

            # Flash = reference is open but test is blocked
            flash_pixels = ref_opening_pixels - test_opening_pixels
            flash_percentage = (flash_pixels / ref_opening_pixels * 100) if ref_opening_pixels > 0 else 0

            # Determine status
            if flash_percentage < 5:
                status = "Good"
            elif flash_percentage < 25:
                status = "Minor Flash"
            else:
                status = "Flash Defect"

            hole_details.append({
                "hole_id": idx + 1,
                "center_x": cx,
                "center_y": cy,
                "area": int(area),
                "flash_pixels": int(flash_pixels),
                "flash_percentage": round(flash_percentage, 2),
                "status": status
            })

        # Sort by hole ID
        hole_details.sort(key=lambda x: x["hole_id"])

        return hole_details

    def _create_labeled_overlay(self, ref_binary: np.ndarray, test_binary: np.ndarray,
                                 flash_mask: np.ndarray, roi_mask: np.ndarray,
                                 hole_details: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization overlay with hole labels.

        Args:
            ref_binary: Reference binary image
            test_binary: Test binary image
            flash_mask: Flash defect mask
            roi_mask: ROI mask
            hole_details: List of hole information

        Returns:
            Labeled overlay image (BGR)
        """
        # Create base overlay
        overlay = self._create_overlay(ref_binary, test_binary, flash_mask, roi_mask)

        # Add hole labels with larger, more visible text
        for hole in hole_details:
            cx, cy = hole["center_x"], hole["center_y"]
            hole_id = hole["hole_id"]
            flash_pct = hole["flash_percentage"]

            # Determine label color based on status (softer, eye-friendly colors)
            if hole["status"] == "Good":
                color = (100, 180, 120)  # Soft teal/green (BGR)
            elif hole["status"] == "Minor Flash":
                color = (100, 180, 255)  # Soft orange/peach (BGR)
            else:
                color = (120, 100, 255)  # Soft red/salmon (BGR)

            # Draw larger circle
            cv2.circle(overlay, (cx, cy), 6, color, -1)

            # Add label with larger font and white background
            label = f"{hole_id}"
            font_scale = 1.0
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get text size for background box
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position text to the right of the circle
            text_x = cx + 10
            text_y = cy + 7

            # Draw white background rectangle with proper wrapping
            # Fix: Account for baseline properly to fully wrap the text
            padding = 4  # Increased padding for better coverage
            box_top = text_y - text_h - baseline - padding
            box_bottom = text_y + baseline + padding
            box_left = text_x - padding
            box_right = text_x + text_w + padding

            cv2.rectangle(overlay,
                         (box_left, box_top),
                         (box_right, box_bottom),
                         (255, 255, 255), -1)

            # Draw text with softer colored outline
            cv2.putText(overlay, label, (text_x, text_y),
                       font, font_scale, color, thickness)

        return overlay

    def _create_overlay(self, ref_binary: np.ndarray, test_binary: np.ndarray,
                        flash_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Create visualization overlay.

        Color coding (showing MATERIAL, not holes):
        - Green: Reference has material (test should also have material here)
        - Red: Test has material (reference should also have material here)
        - Yellow: Both have material (perfect match)
        - Black: Both have openings/holes (also good)
        - Bright Red: Flash defect (reference is open but test has flash material blocking it)

        Note: Binary images have WHITE=openings, BLACK=material
        So we INVERT them to show material in color channels
        """
        h, w = ref_binary.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Apply ROI mask to both binaries
        ref_masked = np.where(roi_mask > 0, ref_binary, 0)
        test_masked = np.where(roi_mask > 0, test_binary, 0)

        # Invert binaries: now 255 = material, 0 = openings
        # This way, color channels represent material presence
        ref_material = 255 - ref_masked
        test_material = 255 - test_masked

        # Reduce intensity for softer, eye-friendly colors
        # Scale from 0-255 to 0-180 for less eye strain
        ref_material = (ref_material * 0.7).astype(np.uint8)  # Max 180 instead of 255
        test_material = (test_material * 0.7).astype(np.uint8)  # Max 180 instead of 255

        # Create base overlay (BGR format in OpenCV)
        overlay[:, :, 1] = ref_material  # Green channel = reference material (softer)
        overlay[:, :, 2] = test_material  # Red channel = test material (softer)

        # Flash defects: reference is OPEN (no material) but test is BLOCKED (has material)
        # Highlight these in softer red that's easier on the eyes
        overlay[flash_mask > 0] = [100, 120, 240]  # BGR: soft red/salmon flash defect

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
