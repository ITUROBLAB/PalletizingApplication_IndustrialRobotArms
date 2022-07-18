import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

import snap7
from snap7.util import *
from snap7.types import *
import time

def ReadMemory(plc, byte, bit, datatype):  # define read memory function
    result = plc.read_area(Areas['MK'], 0, byte, datatype)
    if datatype == S7WLBit:
        return get_bool(result, 0, 1)
    elif datatype == S7WLByte or datatype == S7WLWord:
        return get_int(result, 0)
    elif datatype == S7WLReal:
        return get_real(result, 0)
    elif datatype == S7WLDWord:
        return get_dword(result, 0)
    else:
        return None


def WriteMemory(plc, byte, bit, datatype, value):  # define write memory function
    result = plc.read_area(Areas['MK'], 0, byte, datatype)
    if datatype == S7WLBit:
        set_bool(result, 0, bit, value)
    elif datatype == S7WLByte or datatype == S7WLWord:
        set_int(result, 0, value)
    elif datatype == S7WLReal:
        set_real(result, 0, value)
    elif datatype == S7WLDWord:
        set_dword(result, 0, value)
    plc.write_area(Areas['MK'], 0, byte, result)


def WriteValues(xvall, yvall, thetavall, xadress, yadress, thetaadress):  # define write memory to specific adress function
    WriteMemory(plc, 2, 0, S7WLBit, True)  # write m2.0 True
    WriteMemory(plc, xadress, 0, S7WLReal, xvall)  # write x value to x adress
    WriteMemory(plc, yadress, 0, S7WLReal, yvall)  # write y value to y adress
    WriteMemory(plc, thetaadress, 0, S7WLReal, thetavall)  # write theta value to theta adress
    WriteMemory(plc, 2, 0, S7WLBit, False)  # write m2.0 False


IP = '192.168.0.1'  # IP plc
RACK = 0  # RACK PLC
SLOT = 1  # SLOT PLC

plc = snap7.client.Client()  # call snap7 client function
plc.connect(IP, RACK, SLOT)  # connect to plc

state = plc.get_cpu_state()  # read plc state run/stop/error
print(f'State:{state}')      # print state plc

# Load the image
cap = cv.VideoCapture(0)
x = 0
y = 0
xbuffer = []
ybuffer = []
xsend = 0
ysend = 0

while True:

    ret, frame = cap.read()
    img = frame

    readpermission = ReadMemory(plc, 10, 0, S7WLWord)  # read mw10.0
    WriteMemory(plc, 90, 0, S7WLReal, 1)  # write camera connection value 1

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)


    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 110, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    if readpermission == 1:

        for i, c in enumerate(contours):

            # Calculate the area of each contour
            area = cv.contourArea(c)

            # Ignore contours that are too small or too large
            if area < 25000 or 100000 < area:
                continue

            # cv.minAreaRect returns:
            # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # Retrieve the key parameters of the rotated bounding box
            center = (int(rect[0][0]), int(rect[0][1]))
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = float(rect[2])




            if width < height:
                angle = 90 - angle
            else:
                angle = -angle

            label = "  Rotation Angle: " + str(angle) + " degrees"
            textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
                                   (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
            cv.putText(img, label, (center[0] - 50, center[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
            cv.drawContours(img, [box], 0, (0, 0, 255), 2)

            kx = 0.67757   # calibration for x
            ky = 0.66298   # calibration for y

            x = center[0] * kx
            y = center[1] * ky
            xbuffer.append(x)
            ybuffer.append(y)



            if len(xbuffer) >= 20:

                xsend = sum(xbuffer) / len(xbuffer)
                ysend = sum(ybuffer) / len(ybuffer)

                print("**************************")
                print("X:", xsend)
                print("Y:", ysend)
                print("ANGLE:", angle)
                WriteValues(xsend, ysend, angle, 60, 70, 80)
                xbuffer = []
                ybuffer = []

    else:

        xbuffer = []
        ybuffer = []


    cv.imshow('Output Image', img)
    key = cv.waitKey(1)
    if key == 27:
        break

WriteMemory(plc, 90, 0, S7WLReal, 0)  # write camera connection value 0
cap.release()
cv.destroyAllWindows()


# Save the output image to the current directory
#cv.imwrite("min_area_rec_output.jpg", img)


###########################################################################################################

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("fraame", frame)
#     cv2.waitKey(1)
#
# cap.release()
# cv2.destroyAllWindows()

###########################################################################################################

########################################################## ana yedek:

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# import snap7
# from snap7.util import *
# from snap7.types import *
# import time
#
# def ReadMemory(plc, byte, bit, datatype):  # define read memory function
#     result = plc.read_area(Areas['MK'], 0, byte, datatype)
#     if datatype == S7WLBit:
#         return get_bool(result, 0, 1)
#     elif datatype == S7WLByte or datatype == S7WLWord:
#         return get_int(result, 0)
#     elif datatype == S7WLReal:
#         return get_real(result, 0)
#     elif datatype == S7WLDWord:
#         return get_dword(result, 0)
#     else:
#         return None
#
#
# def WriteMemory(plc, byte, bit, datatype, value):  # define write memory function
#     result = plc.read_area(Areas['MK'], 0, byte, datatype)
#     if datatype == S7WLBit:
#         set_bool(result, 0, bit, value)
#     elif datatype == S7WLByte or datatype == S7WLWord:
#         set_int(result, 0, value)
#     elif datatype == S7WLReal:
#         set_real(result, 0, value)
#     elif datatype == S7WLDWord:
#         set_dword(result, 0, value)
#     plc.write_area(Areas['MK'], 0, byte, result)
#
#
# def WriteValues(xvall, yvall, thetavall, xadress, yadress, thetaadress):  # define write memory to specific adress function
#     WriteMemory(plc, 2, 0, S7WLBit, True)  # write m2.0 True
#     WriteMemory(plc, xadress, 0, S7WLReal, xvall)  # write x value to x adress
#     WriteMemory(plc, yadress, 0, S7WLReal, yvall)  # write y value to y adress
#     WriteMemory(plc, thetaadress, 0, S7WLReal, thetavall)  # write theta value to theta adress
#     WriteMemory(plc, 2, 0, S7WLBit, False)  # write m2.0 False
#
#
# IP = '192.168.0.1'  # IP plc
# RACK = 0  # RACK PLC
# SLOT = 1  # SLOT PLC
#
# plc = snap7.client.Client()  # call snap7 client function
# plc.connect(IP, RACK, SLOT)  # connect to plc
#
# state = plc.get_cpu_state()  # read plc state run/stop/error
# print(f'State:{state}')      # print state plc
#
#
#
# cap = cv2.VideoCapture(0)
# w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print("w: ", w2)
# print("h: ", h2)
#
# # TEMPLATE MATCHING METHOD
# template = cv2.imread("parcatip_ROBLAB.jpg", cv2.IMREAD_GRAYSCALE)
# w, h = template.shape[::-1]
#
# x = 0.0
# y = 0.0
# theta = 11
#
# while True:
#     _, frame = cap.read()
#
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
#     loc = np.where(res >= 0.97)
#
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
#         # Coordinates in mm
#         k = 497/640  # 1px = 0.776mm
#         # print("COORDINATES OF CENTER IN mm: ")
#         # print("x: ", k*(pt[0]+w/2), "mm")
#         # print("y: ", k*(pt[1]+h/2), "mm")
#         x = k * (pt[0] + w / 2)
#         y = k * (pt[1] + h / 2)
#         time.sleep(0.5)
#         print("*********************************************")
#         print("x: ", x, "mm")
#         print("y: ", y, "mm")
#         readpermission = ReadMemory(plc, 10, 0, S7WLWord)  # read mw10.0
#         print('readpermission mw10:', readpermission)
#
#         if readpermission == 1:
#             # write memory
#             WriteValues(x, y, theta, 60, 70, 80)
#             # time.sleep(0.5)
#
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1)
#
#
#
#     if key == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

########################################################################################################## YEDEKKK X Y THETA plc yok

# import cv2 as cv
# from math import atan2, cos, sin, sqrt, pi
# import numpy as np
#
# # Load the image
# cap = cv.VideoCapture(0)
#
#
# x = 0
# y = 0
# xbuffer = []
# ybuffer = []
# xsend = 0
# ysend = 0
#
# while True:
#     ret, frame = cap.read()
#     img = frame
#
#     # Was the image there?
#     if img is None:
#         print("Error: File not found")
#         exit(0)
#
#     # Convert image to grayscale
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#     # Convert image to binary
#     _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#
#     # Find all the contours in the thresholded image
#     contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#
#     for i, c in enumerate(contours):
#
#         # Calculate the area of each contour
#         area = cv.contourArea(c)
#
#         # Ignore contours that are too small or too large
#         if area < 3700 or 100000 < area:
#             continue
#
#         # cv.minAreaRect returns:
#         # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
#         rect = cv.minAreaRect(c)
#         box = cv.boxPoints(rect)
#         box = np.int0(box)
#
#         # Retrieve the key parameters of the rotated bounding box
#         center = (int(rect[0][0]), int(rect[0][1]))
#         width = int(rect[1][0])
#         height = int(rect[1][1])
#         angle = int(rect[2])
#
#         if width < height:
#             angle = 90 - angle
#         else:
#             angle = -angle
#
#         label = "  Rotation Angle: " + str(angle) + " degrees"
#         textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
#                                (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
#         cv.putText(img, label, (center[0] - 50, center[1]),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
#         cv.drawContours(img, [box], 0, (0, 0, 255), 2)
#         kx = 0.7795
#         ky = 0.7826
#         x = center[0]*kx
#         y = center[1]*ky
#         xbuffer.append(x)
#         ybuffer.append(y)
#         xsend = sum(xbuffer) / len(xbuffer)
#         ysend = sum(ybuffer) / len(ybuffer)
#
#         print("X:", x)
#         print("Y:",y)
#         #
#         # if len(xbuffer) >= 70:
#         #     WriteValues(xsend, ysend, theta, 60, 70, 80)
#         #     xbuffer = []
#         #     ybuffer = []
#
#     cv.imshow('Output Image', img)
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cap.release()
# cv.destroyAllWindows()
#
#
# # Save the output image to the current directory
# #cv.imwrite("min_area_rec_output.jpg", img)