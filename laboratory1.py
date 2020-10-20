#Created by Sergei Kochetkov
#Labrab 1 variant 3. MCV

import numpy as np
import cv2
import time


def video_processing():
    #Last frame time value
    prevFrameTime = 0
    #Current frame time
    newFrameTime = 0
    #Type of showing frame. Available values: 1 - x derivative; 2 - y derivative
    # 3 - sum derivative
    showType = 1
    #Declare frame's output depth
    outDepth = cv2.CV_16S
    #Border type declaring
    borderType =  cv2.BORDER_DEFAULT
    #Scale value
    scale = 1
    delta = 0

    #Params for showing FPS and Sobel filter type values
    fpsValuePosition = (10, 50)
    filterTypePosition = (10, 70)
    fontScale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255,255,255)
    lineType = 1

    print("Insert Sobel filter core size...")
    coreSizeStr = input()
    coreSize = int(coreSizeStr)

    print("Starting video processing with Sobel filter core size: ", coreSize)

    cap = cv2.VideoCapture('test.mp4')
    if cap.isOpened():
        window_handle = cv2.namedWindow("Border detection", cv2.WINDOW_AUTOSIZE)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive a frame")
                break

            #Calculate FPS value
            newFrameTime = time.time()
            fps = 1 / (newFrameTime - prevFrameTime)
            prevFrameTime = newFrameTime
            fpsMessage = 'FPS: ' + str(round(fps))
            #GaussianBlur
            frame = cv2.GaussianBlur(frame, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
            #Sobel filter
            x_deriv = cv2.Sobel(frame, outDepth, 1, 0, ksize=coreSize, scale=scale, delta=delta, borderType=borderType)
            y_deriv = cv2.Sobel(frame, outDepth, 0, 1, ksize=coreSize, scale=scale, delta=delta, borderType=borderType)

            x_abs = cv2.convertScaleAbs(x_deriv)
            y_abs =cv2.convertScaleAbs(y_deriv)

            keyCode = cv2.waitKey(33)
            #Check pushed button value
            #Esc button pushed. exit
            if keyCode == 27:
                break
            #Switch Sobel filter mode button
            if keyCode == ord('f'):
                if showType >= 1 and showType < 3:
                    showType += 1
                else:
                    showType = 1

            #Show only x derivative
            if showType == 1:
                resultFrame = x_abs
                modeMessage = "x derivative"

            if showType == 2:
                resultFrame = y_abs
                modeMessage = "y derivative"

            if showType == 3:
                resultFrame = cv2.addWeighted(x_abs, 0.5, y_abs, 0.5, 0)
                modeMessage = "sum of derivatives"

            cv2.putText(resultFrame, modeMessage, filterTypePosition, font, fontScale, fontColor, lineType)
            cv2.putText(resultFrame, fpsMessage, fpsValuePosition, font, fontScale, fontColor, lineType)
            cv2.imshow('Border detection', resultFrame)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Can't open video")


if __name__ == "__main__":
    video_processing()
