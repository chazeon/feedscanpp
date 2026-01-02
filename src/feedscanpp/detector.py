import cv2
import numpy as np
import imutils  # type: ignore
from .geometry import compute_intersection

class DocumentDetector:
    def __init__(self, processing_height=500):
        self.proc_h = processing_height

    def find_corners(self, image_path):
        """
        Returns the 4 corner coordinates (scaled to original image size)
        or None if detection fails.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load {image_path}")
        
        # Keep original for later, resize for detection
        ratio = image.shape[0] / float(self.proc_h)
        small = imutils.resize(image, height=self.proc_h)
        h, w = small.shape[:2]

        # Preprocessing
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5) # Handle scanner noise
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough Lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            print("Warning: No lines detected. Returning full image bounds.")
            return self._get_default_corners(w, h, ratio)

        # Separate Vertical vs Horizontal
        vertical = []
        horizontal = []
        for line in lines:
            rho, theta = line[0]
            if (theta < np.pi/4 or theta > 3*np.pi/4):
                vertical.append(line)
            else:
                horizontal.append(line)

        # Fallback Defaults (Image Edges)
        left_edge   = np.array([[0, 0]])
        right_edge  = np.array([[w, 0]])
        top_edge    = np.array([[0, np.pi/2]])
        bottom_edge = np.array([[h, np.pi/2]])

        # Sort and Pick Best Lines
        # vertical.sort(key=lambda x: x[0][0])
        # horizontal.sort(key=lambda x: x[0][0])

        best_left   = vertical[0] if vertical else left_edge
        best_right  = vertical[1] if len(vertical) > 1 else right_edge
        best_top    = horizontal[0] if horizontal else top_edge
        best_bottom = horizontal[1] if len(horizontal) > 1 else bottom_edge

        # Intersect
        corners = []
        corners.append(compute_intersection(best_top, best_left))     # TL
        corners.append(compute_intersection(best_top, best_right))    # TR
        corners.append(compute_intersection(best_bottom, best_right)) # BR
        corners.append(compute_intersection(best_bottom, best_left))  # BL

        if None in corners:
            return self._get_default_corners(w, h, ratio)

        # Scale corners back to original size
        corners_array = np.array(corners, dtype="float32")
        corners_array *= ratio
        return corners_array

    def _get_default_corners(self, w, h, ratio):
        """ Returns corners of the full image if detection fails """
        corners = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype="float32")
        return corners * ratio