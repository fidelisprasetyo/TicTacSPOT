import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

AREA_MIN = 100

def prepare_image_response(image_responses):
    dtype = np.uint8
    frame = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
    frame = cv2.imdecode(frame, -1)

    if image_responses[0].source.name[0:5] == "front":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    elif image_responses[0].source.name[0:5] == "right":
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    return frame

def convert_to_bin(frame, equalize = True, close = True, erosion = False):

    # Gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # # Histogram Equalization
    if equalize:
        chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frame = chale.apply(frame)

    # Otsu's thresholding
    otsu_t, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing
    if close:
        kernel = np.ones((5, 5), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    if erosion:
        kernel = np.ones((5,5), np.uint8)
        frame = cv2.erode(frame, kernel, iterations=1)

    return frame

def compute_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY

# grid map:
# | [0,0] | [0,1] | [0,2]
# | [1,0] | [1,1] | [1,2]
# | [2,0] | [2,1] | [2,2]

def get_grid(grids, row, col):
    """Returns tuple (contour, (x,y))"""
    if row == 2 and len(grids) < 3:
        print(f"[Warning] Not enough grids for bottom row (found {len(grids)}).")
        return None
    elif row == 1 and len(grids) < 6:
        print(f"[Warning] Not enough grids for middle row (found {len(grids)}).")
        return None
    elif row == 0 and len(grids) < 9:
        print(f"[Warning] Not enough grids for top row (found {len(grids)}).")
        return None
    
    centers_px = [(grid, compute_center(grid)) for grid in grids]
    y_sorted = sorted(centers_px, key=lambda c: c[1][1], reverse=True)

    try:
        # Within each row, sort by X ascending (left to right)
        x_sorted_bottom = sorted(y_sorted[:3], key=lambda c: c[1][0])
        x_sorted_middle = sorted(y_sorted[3:6], key=lambda c: c[1][0])
        x_sorted_top = sorted(y_sorted[6:9], key=lambda c: c[1][0])        

        if row == 0:
            return x_sorted_top[col]
        if row == 1:
            return x_sorted_middle[col]
        if row == 2:
            return x_sorted_bottom[col]
        
    except IndexError:
        print(f"[Error] Column index {col} out of range for row {row}.")
        return None

def get_board_grids(frame, area_min = AREA_MIN):
    """Returns a list of contour of the grids (unsorted) and the contour of board outline"""

    rectangles = defaultdict(list)
    grids = []
    board_outline = None

    bin_frame = convert_to_bin(frame)
    contours, hierarchy = cv2.findContours(bin_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent == -1:
                continue
            cnt_approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            # filter out non-rectangles
            if len(cnt_approx) == 4 and cv2.isContourConvex(cnt_approx):
                area = cv2.contourArea(cnt_approx)
                if area > area_min:
                    rectangles[parent].append(cnt_approx)

    for parent, children in rectangles.items():
        if len(children) >= 3:
            for child in children:
                grids.append(child)
            if len(children) == 9:
                outline_approx = cv2.approxPolyDP(contours[parent], 0.02 * cv2.arcLength(contours[parent], True), True)
                board_outline = outline_approx

    return grids, board_outline

def get_board_outline(frame, approx_area):
    bin_frame = convert_to_bin(frame)
    contours, _ = cv2.findContours(bin_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for _, cnt in enumerate(contours):
        cnt_approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(cnt_approx) == 4 and cv2.isContourConvex(cnt_approx):
            area = cv2.contourArea(cnt_approx)
            if area < approx_area * 1.3 and area > approx_area * 0.7:
                return cnt_approx
    return None
    

def detect_blobs(frame, area):
    bin_frame = convert_to_bin(frame)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = area * 0.3
    params.maxArea = area * 1.7
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(bin_frame)

    return keypoints

def is_px_inside_contour(contour, x, y):
    return cv2.pointPolygonTest(contour, (x, y), False) > 0

def is_x_aligned(x, x_target, threshold):
    return abs(x - x_target) <= threshold

def is_y_aligned(y, y_target, threshold):
    return abs(y - y_target) <= threshold

def draw_board_centers(frame, grids):
    for grid in grids:
        grid_px = compute_center(grid)
        cv2.putText(frame, ".", grid_px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_board(frame, contour, color = (128, 128, 128)):
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, color, 2)

def save_debug_image(image, title="Image", filename="output.png", cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def save_histogram(image, title, filename, otsu_thresh):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(title)
    plt.plot(hist)
    plt.xlabel('intensity')
    plt.ylabel('# of pixels')
    plt.axvline(x=otsu_thresh, color='red', linestyle='--', label=f"Otsu: {int(otsu_thresh)}")
    plt.savefig(filename)
