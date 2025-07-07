import cv2
import numpy as np
import matplotlib.pyplot as plt
from bosdyn.client.image import ImageClient

def detect_contour(frame):
    AREA_MIN = 1000

    gray_frame_overlay = frame.copy()
    # Gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # Histogram Equalization
    chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_frame = chale.apply(blurred_frame)
    # Otsu's thresholding
    otsu_t, bin_frame = cv2.threshold(eq_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, hierarchy = cv2.findContours(closed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_count = 0
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > AREA_MIN and hierarchy[0][idx][3] != -1:
                    cv2.drawContours(gray_frame_overlay, [approx], -1, (255, 0, 0), 2)
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(gray_frame_overlay, f"{rectangle_count+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    rectangle_count += 1
    
    # Show result
    cv2.imshow("Tictacspot_gray", gray_frame_overlay)
    cv2.imshow("Tictacspot_bin", closed_frame)

def track(image_client: ImageClient, image_source = "right_fisheye_image"):

    AREA_MAX = 1000

    while True:
        image_responses = image_client.get_image_from_sources([image_source])

        dtype = np.uint8

        gray_frame = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        gray_frame = cv2.imdecode(gray_frame, -1)

        if image_responses[0].source.name[0:5] == "front":
            gray_frame = cv2.rotate(gray_frame, cv2.ROTATE_90_CLOCKWISE)

        elif image_responses[0].source.name[0:5] == "right":
            gray_frame = cv2.rotate(gray_frame, cv2.ROTATE_180)

        gray_frame = cv2.resize(gray_frame, (640, 480))

        gray_frame_overlay = gray_frame.copy()

        # Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Histogram equalization
        chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq_frame = chale.apply(blurred_frame)

        # Otsu thresholding / Adaptive thresholding
        otsu_t, bin_frame = cv2.threshold(eq_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)
        closed_frame = cv2.morphologyEx(bin_frame, cv2.MORPH_CLOSE, kernel)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(closed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw and label detected rectangles
        rectangle_count = 0
        if hierarchy is not None:
            for idx, cnt in enumerate(contours):
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if area > AREA_MAX and hierarchy[0][idx][3] != -1:
                        cv2.drawContours(gray_frame_overlay, [approx], -1, (255, 0, 0), 2)
                        M = cv2.moments(approx)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(gray_frame_overlay, f"{area}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        rectangle_count += 1

        # Show result
        cv2.imshow("Tictacspot", gray_frame_overlay)
        cv2.imshow("Tictacspot_bin", closed_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

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


# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open Camera")
#     exit()

# while True:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     detect_contour(frame)

#         # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

