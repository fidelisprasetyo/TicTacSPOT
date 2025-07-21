import cv2
import numpy as np
import matplotlib.pyplot as plt
from bosdyn.client.image import ImageClient
from collections import defaultdict

AREA_MIN = 200

def prepare_image_response(image_responses):
    dtype = np.uint8
    frame = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
    frame = cv2.imdecode(frame, -1)

    if image_responses[0].source.name[0:5] == "front":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    elif image_responses[0].source.name[0:5] == "right":
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    return frame

def convert_to_bin(frame):

    # Gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # # Histogram Equalization -> too heavy
    # chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # eq_frame = chale.apply(blurred_frame)
    # Otsu's thresholding
    otsu_t, bin_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, kernel)

    return closed_frame

def compute_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY

# won't be used
def draw_board(frame, grids):
    for contour in grids:
        cv2.drawContours(frame, [contour], -1, (125, 125, 125), 2)

# grid tuple format: ((x,y), area)
# grid map:
# | (2,0) | (2,1) | (2,2)
# | (1,0) | (1,1) | (1,2)
# | (0,0) | (0,1) | (0,2)

def get_grid(grids, row, col):

    if row == 0 and len(grids) < 3:
        print(f"[Warning] Not enough grids for bottom row (found {len(grids)}).")
        return None
    elif row == 1 and len(grids) < 6:
        print(f"[Warning] Not enough grids for middle row (found {len(grids)}).")
        return None
    elif row == 2 and len(grids) < 9:
        print(f"[Warning] Not enough grids for top row (found {len(grids)}).")
        return None

    y_sorted = sorted(grids, key=lambda x: x[0][1], reverse=True)

    try:
        x_sorted_bottom = sorted(y_sorted[:3], key=lambda x: x[0][0])
        x_sorted_middle = sorted(y_sorted[3:6], key=lambda x: x[0][0])
        x_sorted_top = sorted(y_sorted[6:9], key=lambda x: x[0][0])
        if row == 0:
            return x_sorted_bottom[col]
        if row == 1:
            return x_sorted_middle[col]
        if row == 2:
            return x_sorted_top[col]

    except IndexError:
        print(f"[Error] Column index {col} out of range for row {row}.")
        return None


def draw_board_centers(frame, grids):
    for grid in grids:
        point = grid[0]
        cv2.putText(frame, ".", point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def get_board_grids(frame):

    rectangles = defaultdict(list)
    grids = []

    bin_frame = convert_to_bin(frame)

    # Find contours
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
                if area > AREA_MIN:
                    rectangles[parent].append(cnt_approx)

    for parent, children in rectangles.items():
        if len(children) >= 3:
            for child in children:
                grid_tuple = (compute_center(child), cv2.contourArea(child))
                grids.append(grid_tuple)

    return grids

def is_x_aligned(grid, threshold, x):
    x_grid = grid[0][0]
    return abs(x_grid - x) <= threshold

def is_y_aligned(grid, threshold, y):
    y_grid = grid[0][1]
    return abs(y_grid - y) <= threshold

def area_diff(grid, target_area):
    return abs(grid[1] - target_area)

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
