import cv2
import numpy as np

def detect_color_mode(image, thumb_size=500):
    """
    Robustly detects: 'Color', 'Grayscale', 'Black & White', or 'Sepia'.
    Handles sparse color (like a single red title) by checking Vividness.
    """
    h, w = image.shape[:2]
    scale = thumb_size / max(h, w)
    if scale < 1:
        thumb = cv2.resize(image, None, fx=scale, fy=scale)  # type: ignore[arg-type]
    else:
        thumb = image

    # 1. HSV Analysis
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # --- STEP 1: DETECT COLOR (Sparse vs Full) ---

    # "Potential Color": Any pixel with Saturation > 25 (0-255 scale).
    # This filters out gray paper and white background.
    color_mask = (sat > 25)
    color_pixels = np.count_nonzero(color_mask)
    color_ratio = color_pixels / thumb.size

    if color_ratio > 0:
        # Calculate the AVERAGE saturation of only the "colored" pixels.
        # This tells us: "Is the color we found Vivid (Ink) or Dull (Noise)?"
        avg_saturation = np.mean(sat[color_mask])
    else:
        avg_saturation = 0

    # CASE A: Full Color Page (Photos, Sepia, Advertisement)
    # If > 2% of the page is colored, it's definitely not B&W.
    if color_ratio > 0.02:
        # Sepia Check: High coverage, but low variance in Hue (all yellow/brown)
        if np.std(hue[color_mask]) < 20:
            return "Grayscale"  # Treat Sepia as Grayscale for simplicity
        return "Color"

    # CASE B: Sparse Color (Red Title, Blue Signature, Stamp)
    # The ratio is small (0.02% to 2%), but the "Vividness" (Avg Saturation) is high.
    # Real Ink (Red/Blue) usually has Saturation > 80.
    # Sensor Noise usually has Saturation < 40.
    elif color_ratio > 0.0005: # threshold: 0.05% of page (very sensitive)
        if avg_saturation > 60:
            return "Color" # Small vivid text found!

    # --- STEP 2: DETECT B&W vs GRAYSCALE ---

    # If we are here, the image is effectively Monochromatic.
    # We check the HISTOGRAM of the Value channel.

    hist = cv2.calcHist([val], [0], None, [256], [0, 256])
    
    # A clean B&W doc has two spikes: Ink (0-50) and Paper (200-255).
    # A Grayscale photo has a "hill" in the middle (50-200).
    
    # Count pixels in the "Mid-Tone" range.
    middle_grays = np.sum(hist[50:200])
    middle_ratio = middle_grays / thumb.size

    # If > 10% of pixels are mid-tones, it's likely a Grayscale Photo/Art.
    # (Standard text docs usually have < 5% anti-aliasing pixels)
    if middle_ratio > 0.10:
        return "Grayscale"
        
    return "Black & White"

def remove_tint(image):
    # 1. Convert to HSV to isolate "Value" (Brightness)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 2. Estimate the background tint using the Value channel
    # Using a larger kernel for dilation to ensure we cover large text/stamps
    kernel = np.ones((41, 41), np.uint8)
    dilated = cv2.dilate(v, kernel)
    
    # Use a large Gaussian blur instead of Median for smoother transitions
    bg_estimate = cv2.GaussianBlur(dilated, (41, 41), 0)

    # 3. Division Normalization (v / bg_estimate)
    # This turns the background (paper) into pure white (255)
    # We use float32 to avoid rounding errors during math
    v_norm = cv2.divide(v, bg_estimate, scale=255)

    # 4. Merge back and convert to BGR
    final_hsv = cv2.merge([h, s, v_norm])
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # 5. Local Contrast Enhancement (Optional but replaces your Gamma logic)
    # CLAHE works much better than a global Gamma for making ink "pop"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return result

def enhance_image(image, mode: str):
    """
    Routes the image to the correct mathematical cleaning pipeline.
    """
    if mode == "Color":
        return image
        
    # Convert to gray for all other modes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if mode == "Black & White":
        # Already clean? Just Otsu it to be sure.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    return gray
