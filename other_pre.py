import cv2
import numpy as np
from PIL import Image

# 读取一张图片
imgPath = "data-raw/1-Images/1-Training Set/00000ce7.png"
image = cv2.imread(imgPath)

# 显示原始图像
cv2.imshow('Original Image', image)

# 取绿色通道
r, imageGreen, b = cv2.split(image)
#cv2.imshow('Green Channel Image', imageGreen)

# 对绿色通道进行对比度有限的自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imageEqualized = clahe.apply(imageGreen)
#cv2.imshow('Histogram Equalized Image', imageEqualized)

# 反转处理
imageInv2 = 255 - imageEqualized
imageInv = clahe.apply(imageInv2)
#cv2.imshow('Inverted Image', imageInv)

# 中值滤波器去除噪声
#imageMed = cv2.medianBlur(imageInv, 5)
#cv2.imshow('Median Filtered Image', imageMed)
imageMed = imageInv

# 顶帽操作去除背景
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
imageOpen = cv2.morphologyEx(imageMed, cv2.MORPH_OPEN, kernel)
imageBackElm = imageMed - imageOpen
cv2.imshow('Background Eliminated Image', imageBackElm)

# 自适应阈值处理
#imagethresh2 = cv2.adaptiveThreshold(imageBackElm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)
#cv2.imshow('Adaptive Thresholded Image', imagethresh2)

# 等待按键
cv2.waitKey(0)
cv2.destroyAllWindows()
