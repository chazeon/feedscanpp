import numpy as np

def compute_intersection(line1, line2):
    """
    Finds the (x, y) intersection of two lines given in (rho, theta).
    Returns None if lines are parallel.
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    try:
        x0, y0 = np.linalg.solve(A, b)
        return [int(np.round(x0)), int(np.round(y0))]
    except np.linalg.LinAlgError:
        return None

def order_points(pts):
    """
    Orders a list of 4 coordinates: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    
    return rect