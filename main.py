import cv2
import numpy as np
from rembg import remove


def automatic_object_border_detection (img_path):
    image = cv2.imread(img_path)

    if len(image.shape) > 2 and image.shape[2] == 4:
        image = image[:, :, :3]

    cv2.namedWindow("Resized_Window", 2)
    cv2.resizeWindow("Resized_Window", 1200, 1000)

    r = cv2.selectROI("Resized_Window", image, False, False)

    cropped_image = image[int(r[1]):int(r[1]+r[3]),
                            int(r[0]):int(r[0]+r[2])]

    output = remove(cropped_image)

    if len(output.shape) > 2 and output.shape[2] == 4:
        output = output[:, :, :3]

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset = (int(r[0]/2), int(r[1]/2)))


    for cont in contours:
        # if cv2.contourArea(cont) < 190000:
            # continue
        # epsilon = 0.05 * cv2.arcLength(cont, True)
        # approx = cv2.approxPolyDP(cont, epsilon, True)

        cv2.drawContours(image, [cont], -1, (255, 0, 0), 7, offset = (int(r[0]/2), int(r[1]/2)))

    cv2.imshow("Resized_Window", image)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


img_path = './images/4.jpg'
automatic_object_border_detection(img_path)
