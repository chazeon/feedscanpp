import cv2
import numpy as np
import imutils
from scipy.optimize import minimize_scalar

def detect_skew_angle(image):
    """
    Detects the skew angle of an image using the 4th harmonic of Hough Lines.
    Returns the angle in degrees needed to rotate the image to be upright.
    """

    small_img = imutils.resize(image, width=600)

    gray = 255 - cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    # 1. Determine the largest dimension to make it a square
    h, w = gray.shape
    side = max(cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w))

    # 2. Pad to a SQUARE (side x side)
    pad_y = side - h
    pad_x = side - w

    # Center the image in the square to keep the FFT "beam" centered
    top, bottom = pad_y // 2, pad_y - (pad_y // 2)
    left, right = pad_x // 2, pad_x - (pad_x // 2)

    padded = cv2.copyMakeBorder(gray, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=0)
    
    padded = cv2.GaussianBlur(padded, (7, 7), 0)

    window_y = np.hanning(side)
    window_x = np.hanning(side)
    window = np.outer(window_y, window_x)
    padded = (padded.astype(float) * window).astype(np.float32)

    # 2. Compute FFT
    dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 3. Calculate Magnitude Spectrum
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

    def get_radial_score_masked(angle, mag_spectrum):
        h, w = mag_spectrum.shape
        center = (w // 2, h // 2)
        
        # 1. Rotate the spectrum
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(mag_spectrum, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # 2. Apply a circular mask to ignore the dark 'corners' created by rotation
        # This ensures the score is based ONLY on real FFT data
        mask = np.zeros((h, w), dtype=np.uint8)
        radius = min(h, w) // 3  # Use the central 1/3rd where signal is strongest
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask
        valid_region = cv2.bitwise_and(rotated, rotated, mask=mask)

        # 3. Robust Scoring: Sum variance of horizontal projections
        # We sum rows but only within the central mask region
        # To do this correctly, we take the sum and then only use the central indices
        row_sums = np.sum(valid_region, axis=1)
        col_sums = np.sum(valid_region, axis=0)
        # Variance is maximized when parallel text strips align
        return -np.std(col_sums) ** 8 - np.std(row_sums) ** 8
    # 4. Optimization Search
    # The FFT beam is perpendicular to text. Search range (-45 to 45).
    # We use Brent's method to find the peak energy angle.
    res = minimize_scalar(get_radial_score_masked, args=(magnitude,), 
                            bounds=(-5, 5), method='bounded', options={'xatol': 0.001})

    # The beam angle is perpendicular to the skew angle
    skew_angle = res.x

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
