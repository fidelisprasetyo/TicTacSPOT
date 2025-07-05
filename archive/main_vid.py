import cv2
import numpy as np
import matplotlib.pyplot as plt

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

debug_capture = False
use_otsu = True
equalize = True

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open Camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_overlay = gray_frame.copy()

    # Gaussian blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Histogram equalization
    if(equalize):
        chale = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq_frame = chale.apply(blurred_frame)
    else:
        eq_frame = blurred_frame

    # if(debug_capture):
    #     otsu_t_blur, temp_bin_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Otsu thresholding / Adaptive thresholding
    if(use_otsu):
        otsu_t, bin_frame = cv2.threshold(eq_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        bin_frame = cv2.adaptiveThreshold(eq_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

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
                if area > 10000 and hierarchy[0][idx][3] != -1:
                    cv2.drawContours(gray_frame_overlay, [approx], -1, (255, 0, 0), 2)
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(gray_frame_overlay, f"{rectangle_count+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    rectangle_count += 1

    # Show result
    cv2.imshow("Tictacspot", gray_frame_overlay)

    if debug_capture:
        save_debug_image(gray_frame, "grayscale", "input.png")
        save_debug_image(blurred_frame, "blurred", "blurred.png")
        save_debug_image(eq_frame, "equalized", "equalized.png" )
        #save_histogram(blurred_frame, "before equalization", "blurred_hist.png", otsu_t_blur)
        save_histogram(eq_frame, "histogram", "hist.png", otsu_t)
        save_debug_image(bin_frame, "binary img", "binary_img.png")
        save_debug_image(closed_frame, "closed_img", "closed_img.png")

        contour_preview = cv2.cvtColor(gray_frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_preview, contours, -1, (0, 255, 0), 1)
        save_debug_image(contour_preview, "contours", "contours.png")
        save_debug_image(gray_frame_overlay, "final", "output.png")

        debug_capture = False

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()