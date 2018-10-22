import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import sys
import math

cam = cv2.VideoCapture(1)

openni2.initialize("F:\handTracking\OpenNI-Windows-x64-2.3\Redist")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps=30))
while True:
    _, oriImg = cam.read()

    if oriImg is None:
        print('oriImg is None')
        sys.exit(-1)

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (1, 480, 640)
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    img_uint8 = np.array(img / 256, dtype=np.uint8)
    # img_uint8 = cv2.flip(img_uint8, 1)

    img_uint8_max = float(np.amax(img_uint8) / 256.0)

    processed = cv2.add(img_uint8, img_uint8)
    processed = cv2.add(processed, img_uint8)
    processed = cv2.add(processed, img_uint8)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 0.0, cv2.THRESH_TOZERO)

    # Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    hand = cv2.subtract(gray, sure_bg)

    dst = cv2.medianBlur(hand, 9)
    '''
    dst.shape = (1, 480, 640)
    dst = np.concatenate((dst, dst, dst), axis=0)
    dst = np.swapaxes(dst, 0, 2)
    dst = np.swapaxes(dst, 0, 1)

    scale = 368 / (oriImg.shape[0] * 1.0)
    
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    d_imageToTest = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((368, 368, 3)) * 128
    processed_img = np.ones((boxsize, boxsize, 3)) * 128
    depth_img = np.ones((368, 368, 3)) * 128
    
    img_w = imageToTest.shape[1]
    if img_w < 368:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(368 / 2 - math.floor(img_w / 2)):int(
            368 / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
        # processed_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
        # boxsize / 2 + math.floor(img_w / 2) + offset), :] = d_imageToTest
        depth_img[:, int(368 / 2 - math.floor(img_w / 2)):int(
            368 / 2 + math.floor(img_w / 2) + offset), :] = d_imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - 368 / 2):int(img_w / 2 + 368 / 2), :]
        # processed_img = d_imageToTest[:,
        # int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
        depth_img = d_imageToTest[:,
                    int(img_w / 2 - 368 / 2):int(img_w / 2 + 368 / 2), :]
    '''
    scale = 368 / (oriImg.shape[0] * 1.0)
    hand_resized = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    ret, hand_mask = cv2.threshold(hand_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hand_mask = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
    hand_mask = cv2.flip(hand_mask, 1)

    # mask의 범위 확장
    kernel = np.ones((5, 5), np.uint8)
    hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)

    print(hand_mask.shape)

    check_nonzero = cv2.findNonZero(hand_mask[:, :, 0])
    count_nonzero = cv2.countNonZero(hand_mask[:, :, 0])
    if count_nonzero > 0:
        check_nonzero_av_x = np.average(check_nonzero[0])
        check_nonzero_av_y = np.average(check_nonzero[1])
        check_nonzero_av_x = float(check_nonzero_av_x - 245.5) / 491.0
        check_nonzero_av_y = float(check_nonzero_av_y - 184.0) / 368.0
        print(check_nonzero_av_x)
        print(check_nonzero_av_y)

    # cv2.imshow("Hand_mask", hand_mask)

    h, w = hand_mask.shape[:2]
    print(img_uint8_max)
    width_cal = 0.0
    height_cal = 0.0
    if count_nonzero > 0.0:
        width_cal = -27.5 - 85.0 * check_nonzero_av_x + 20.0 * check_nonzero_av_y
        if check_nonzero_av_x < -0.2:
            width_cal -= 25.0 * check_nonzero_av_x
        elif check_nonzero_av_x > 0.0:
            width_cal -= 110.0 * check_nonzero_av_x
        height_cal = 5.0 - 95.0 * check_nonzero_av_y + 95.0 * check_nonzero_av_x
        if check_nonzero_av_y > 0.0:
            height_cal -= 120.0 * check_nonzero_av_y
        M = np.float32([[1, 0, width_cal], [0, 1, height_cal]])
    hand_mask = cv2.warpAffine(hand_mask, M, (w, h))

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    # imageToTest = cv2.bitwise_and(hand_mask, imageToTest)
    d_imageToTest = cv2.resize(hand_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((368, 368, 3)) * 128
    d_output_img = np.ones((368, 368, 3)) * 128

    img_h = imageToTest.shape[0]
    img_w = imageToTest.shape[1]
    if img_w < 368:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(368 / 2 - math.floor(img_w / 2)):int(
            368 / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
        d_output_img[:, int(368 / 2 - math.floor(img_w / 2)):int(
            368 / 2 + math.floor(img_w / 2) + offset), :] = d_imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - 368 / 2):int(img_w / 2 + 368 / 2), :]
        d_output_img = d_imageToTest[:,
                     int(img_w / 2 - 368 / 2):int(img_w / 2 + 368 / 2), :]

    # depth_float_img = np.array(depth_img / 256, dtype=np.float)
    # processed_img = np.array(output_img * depth_float_img, dtype=np.uint8)
    cv2.flip(output_img, 1)
    cv2.imshow("depth", d_output_img)
    cv2.imshow("image", output_img)
    # cv2.imshow("depth", depth_float_img)

    if cv2.waitKey(1) == ord('q'): break
openni2.unload()