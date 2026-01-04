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
    """
    Removes yellow/gray background -> White.
    Keeps Red/Blue/Green ink vivid.
    """
    # 1. Split into 3 channels (Blue, Green, Red)
    planes = cv2.split(image)
    norm_planes = []

    # 2. Process each color channel independently
    for plane in planes:
        # --- A. Estimate Background ---
        # Dilate to find the paper color (ignoring the ink)
        dilated = cv2.dilate(plane, np.ones((25, 25), np.uint8))
        
        # Blur to smooth the estimate
        bg_estimate = cv2.medianBlur(dilated, 21)
        
        # --- B. Normalize (Divide) ---
        # Result = (Plane / Background) * 255
        # This forces the background to 255 (White) in this specific color channel
        normalized = cv2.divide(plane, bg_estimate, scale=255)
        
        norm_planes.append(normalized)

    # 3. Merge back into a Color Image
    result = cv2.merge(norm_planes)  # type: ignore[arg-type]

    # 4. Level Adjustment (Optional but Recommended)
    # This makes the ink darker and crisper, just like your example.
    # We apply a slight gamma correction to all channels.
    invGamma = 0.95  # 1.0 = No change. Lower (0.8-0.9) = Darker/Vivid ink.
    
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    final_result = cv2.LUT(result, table)

    return final_result

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