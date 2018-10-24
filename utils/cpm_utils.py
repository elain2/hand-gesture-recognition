import numpy as np
import math
import cv2


M_PI = 3.14159


# Compute gaussian kernel for input image
def gaussian_img(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def read_image(cam, boxsize, dep_stream):
    # from file
    _, oriImg = cam.read()

    if oriImg is None:
        print('oriImg is None')
        return None

    # Get Depth map
    frame = dep_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(frame_data, dtype=np.uint16)
    depth_img.shape = (1, 480, 640)

    depth_img_uint8 = np.array(depth_img / 256, dtype=np.uint8)

    depth_img_uint8 = np.concatenate((depth_img_uint8, depth_img_uint8, depth_img_uint8), axis=0)
    depth_img_uint8 = np.swapaxes(depth_img_uint8, 0, 2)
    depth_img_uint8 = np.swapaxes(depth_img_uint8, 0, 1)
    depth_img_uint8 = cv2.flip(depth_img_uint8, 1)

    # Depth map Translation
    h, w = depth_img_uint8.shape[:2]
    width_cal = -28
    height_cal = -7
    M = np.float32([[1, 0, width_cal], [0, 1, height_cal]])
    depth_img_uint8 = cv2.warpAffine(depth_img_uint8, M, (w, h))

    # Depth map processing
    processed = cv2.add(depth_img_uint8, depth_img_uint8)
    processed = cv2.add(processed, depth_img_uint8)
    processed = cv2.add(processed, depth_img_uint8)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 0.0, cv2.THRESH_TOZERO)

    # make hand mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    hand = cv2.subtract(gray, sure_bg)
    dst = cv2.medianBlur(hand, 9)
    scale = boxsize / (oriImg.shape[0] * 1.0)
    hand_resized = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    ret, hand_mask_gray = cv2.threshold(hand_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hand_mask = cv2.cvtColor(hand_mask_gray, cv2.COLOR_GRAY2BGR)

    # mask의 범위 확장
    kernel = np.ones((5, 5), np.uint8)
    hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)

    h, w = hand_mask.shape[:2]
    width_cal = -14.5
    height_cal = -5.0
    M = np.float32([[1, 0, width_cal], [0, 1, height_cal]])

    hand_mask = cv2.warpAffine(hand_mask, M, (w, h))
    depth_img_uint8 = cv2.warpAffine(depth_img_uint8, M, (w, h))

    # hand_mask.shape = (1, 480, 640)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    d_imageToTest = cv2.resize(depth_img_uint8, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # alpha blending
    hand_mask_inv = cv2.bitwise_not(hand_mask)
    alpha = np.array(hand_mask_inv / 255, dtype=float)
    background = np.array(imageToTest, dtype=float)
    foreground = np.array(imageToTest * 0.4, dtype=float)
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    imageToTest_process = cv2.add(foreground, background)
    imageToTest_process = np.array(imageToTest_process, dtype=np.uint8)

    # Get Skeleton from Depth
    b, g, r = cv2.split(hand_mask)
    tmp = np.ones(b.shape)
    tmp = tmp * (-20)
    b -= np.uint8(tmp)
    hand_mask = cv2.merge([b, g, r])

    output_img = np.ones((boxsize, boxsize, 3)) * 128
    processed_img = np.ones((boxsize, boxsize, 3)) * 128
    depth_img = np.ones((boxsize, boxsize, 3)) * 128

    img_w = imageToTest.shape[1]
    if img_w < boxsize:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
        processed_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest_process
        depth_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = d_imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
        processed_img = imageToTest_process[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
        depth_img = d_imageToTest[:,
                        int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]


    return output_img, processed_img, depth_img