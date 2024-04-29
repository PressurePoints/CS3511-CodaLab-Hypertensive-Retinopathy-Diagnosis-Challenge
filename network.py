import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 40 * 40, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 40 * 40)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def preprocess_image(image):
    # 取绿色通道
    r, imageGreen, b = cv2.split(image)
    #cv2.imshow('Green Channel Image', imageGreen)

    # 对绿色通道进行对比度有限的自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imageEqualized = clahe.apply(imageGreen)
    
    imageEqualized_r = clahe.apply(r)
    imageEqualized_b = clahe.apply(b)
    imageEqualized_rgb = cv2.merge([imageEqualized_r, imageEqualized, imageEqualized_b])
    #cv2.imshow('Histogram Equalized Image', imageEqualized)

    # 反转处理
    #imageInv2 = 255 - imageEqualized
    #imageInv = clahe.apply(imageInv2)
    #cv2.imshow('Inverted Image', imageInv)

    # 中值滤波器去除噪声
    #imageMed = cv2.medianBlur(imageInv, 5)
    #cv2.imshow('Median Filtered Image', imageMed)
    #imageMed = imageInv

    # 顶帽操作去除背景
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #imageOpen = cv2.morphologyEx(imageMed, cv2.MORPH_OPEN, kernel)
    #imageBackElm = imageMed - imageOpen
    #imageOpen = cv2.morphologyEx(imageInv, cv2.MORPH_OPEN, kernel)
    #imageBackElm = imageInv - imageOpen
    # 将 numpy 数组转换为 PIL 图像
    #image_3d = np.repeat(imageEqualized[:, :, np.newaxis], 3, axis=2)
    img_pil = Image.fromarray(imageEqualized_rgb)
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img_pil)
