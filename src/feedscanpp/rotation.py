import cv2
import numpy as np
import imutils

def detect_skew_angle(image):
    # Resize for speed
    resizing_factor = 800.0 / image.shape[0]
    small_img = imutils.resize(image, height=800)
    
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    # Use a slightly more adaptive Canny or a blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Standard Hough Lines provides (rho, theta)
    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi/180, 100)
    
    if lines is None:
        return 0.0

    lines = lines[:80] # Take top 80 strongest lines
    
    # Extract theta (index 1) and accumulator votes (index 2)
    angles = lines[:, 0, 1]
    weights = lines[:, 0, 2].astype(float) ** 2

    # Map to 4th harmonic for 90-degree symmetry
    # We subtract pi/2 if we want 0 to represent "upright" 
    # but the 4x trick usually handles the wrapping.
    x = np.cos(4 * angles)
    y = np.sin(4 * angles)

    sum_x = np.sum(x * weights)
    sum_y = np.sum(y * weights)

    combined_angle_4theta = np.arctan2(sum_y, sum_x)
    # This gives us the dominant angle in a 90-degree range
    refined_angle_rad = combined_angle_4theta / 4
    
    # Convert to degrees
    refined_angle_deg = np.rad2deg(refined_angle_rad)

    # IMPORTANT: Hough theta is measured from the vertical axis.
    # To get the rotation angle to straighten the image:
    # If the dominant line is vertical (0 rad), rotation should be 0.
    # If the dominant line is horizontal (pi/2 rad), rotation should be 0.
    
    # Adjusting the output:
    if refined_angle_deg > 45:
        return refined_angle_deg - 90
    elif refined_angle_deg < -45:
        return refined_angle_deg + 90
        
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
