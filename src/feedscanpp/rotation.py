import cv2
import numpy as np
import imutils  # type: ignore

def detect_skew_angle(image):
    """
    Detects the skew angle of the document in the image using Hough Transform
    and circular statistics for sub-degree accuracy.

    Returns the angle in degrees to rotate the image to correct orientation.
    """
    image = imutils.resize(image, height=800)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi/180, 100)[:80]
    if lines is None:
        return 0.0

    # Extract angles (theta) from HoughLines
    angles = lines[:, :, 1].flatten()
    weights = lines[:, :, 2].flatten() ** 2

    # Map angles to 90-degree periodicity
    # Since grids have 4-fold symmetry, we multiply by 4
    # This maps 0°, 90°, 180°, 270° all to the same vector direction
    x = np.cos(4 * angles)
    y = np.sin(4 * angles)

    # Mask out large angles by setting their weights to zero
    mask = np.abs(x) < 0.2
    weights[~mask] = 0.0

    # Calculate the weighted center of mass of the vectors
    sum_x = np.sum(x * weights)
    sum_y = np.sum(y * weights)

    # Convert back to an angle
    combined_angle_4theta = np.arctan2(sum_y, sum_x)
    refined_angle_rad = combined_angle_4theta / 4
    refined_angle_deg = np.rad2deg(refined_angle_rad)

    return refined_angle_deg

def rotate_image(image, angle):
    """
    Rotates the image by the specified angle to correct skew.
    Uses cubic interpolation and border replication for quality.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
