import cv2
import numpy as np

def trim_image(image):
    """
    Trims white borders from the image by finding the bounding box
    of all content (non-white pixels).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image (invert so white becomes black and content becomes white)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Find all non-zero points (the content)
    points = cv2.findNonZero(thresh)

    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(points)

    # Crop the original image
    cropped = image[y:y+h, x:x+w]

    return cropped
