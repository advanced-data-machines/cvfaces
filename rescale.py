import cv2 as cv
import numpy as np


def blank_me(width=500, height=500, channels=3):
    blank = np.zeros((width, height,channels), dtype='uint8')
    return blank

# scaling 
# Images, Video and Live Video supported
def rescaleFrame(frame, scale=0.75):
    width = int( frame.shape[1] * scale)
    height = int( frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# only live video
def changeRes(capture, width, height):
    capture.set(3,width)
    capture.set(4,height)



# -x left
# -y up
# x right
# y down
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

