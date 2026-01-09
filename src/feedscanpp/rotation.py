import cv2
import numpy as np
import imutils

def detect_skew_angle(image):
    """
    Detects the skew angle of an image using the 4th harmonic of Hough Lines.
    Returns the angle in degrees needed to rotate the image to be upright.
    """
    # 1. Preprocessing for speed and noise reduction
    # We resize to a standard height to make the Hough parameters more consistent
    small_img = imutils.resize(image, height=800)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # Blur helps merge text characters into "lines" for better detection
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # 2. Standard Hough Transform
    # rho=1, theta=1 degree, threshold=100
    lines = cv2.HoughLinesWithAccumulator(edges, 1, np.pi/180, 100)
    
    if lines is None:
        return 0.0

    # Limit to the top 80 strongest lines to avoid processing noise
    lines = lines[:80] 
    
    # Extract theta (index 1) and weights (accumulator votes, index 2)
    angles = lines[:, 0, 1]
    weights = lines[:, 0, 2].astype(float) ** 2 # Square weights to prioritize stronger lines

    # 3. 4th Harmonic Mapping
    # This treats 0, 90, 180, and 270 degrees as the same orientation.
    # It solves the problem of "horizontal vs vertical" ambiguity.
    sum_x = np.sum(np.cos(4 * angles) * weights)
    sum_y = np.sum(np.sin(4 * angles) * weights)

    # Result is in range [-pi, pi]
    combined_angle_4theta = np.arctan2(sum_y, sum_x)
    
    # Convert back to actual angle in degrees [-45, 45]
    refined_angle_deg = np.rad2deg(combined_angle_4theta / 4)

    # 4. Final Normalization
    # Since OpenCV Hough 0 is vertical and 90 is horizontal, 
    # we normalize the output to ensure the smallest rotation is chosen.
    if refined_angle_deg > 45:
        skew_angle = refined_angle_deg - 90
    elif refined_angle_deg < -45:
        skew_angle = refined_angle_deg + 90
    else:
        skew_angle = refined_angle_deg

    # Return the inverse to use directly in cv2.getRotationMatrix2D
    return skew_angle

def rotate_image(image, angle):
    """
    Rotates the image by the specified angle.
    A positive angle rotates counter-clockwise.
    """
    if angle == 0:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    # BORDER_REPLICATE prevents black edges from appearing on the corners
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

# --- Example Usage ---
if __name__ == "__main__":
    img = cv2.imread('skewed_document.jpg')
    
    if img is not None:
        angle = detect_skew_angle(img)
        print(f"Detected Skew: {angle:.2f} degrees")
        
        # Correction: We rotate by the detected angle. 
        # If detected skew is 5 degrees (tilted right), 
        # rotate_image will rotate it 5 degrees CCW to fix it.
        corrected_img = rotate_image(img, angle)
        
        cv2.imshow("Original", imutils.resize(img, height=600))
        cv2.imshow("Corrected", imutils.resize(corrected_img, height=600))
        cv2.waitKey(0)
