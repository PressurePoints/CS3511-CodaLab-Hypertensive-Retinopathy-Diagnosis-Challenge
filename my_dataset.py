from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # Load the image
        img = cv2.imread(self.images_path[item])

        # 取绿色通道
        r, imageGreen, b = cv2.split(img)

        # 对绿色通道进行对比度有限的自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imageEqualized = clahe.apply(imageGreen)

        # 反转处理
        imageInv2 = 255 - imageEqualized
        imageInv = clahe.apply(imageInv2)

        # # 中值滤波器去除噪声
        # # imageMed = cv2.medianBlur(imageInv, 5)
        # imageMed = imageInv

        # # 顶帽操作去除背景
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # imageOpen = cv2.morphologyEx(imageMed, cv2.MORPH_OPEN, kernel)
        # imageBackElm = imageMed - imageOpen

        # Convert the images to RGB format
        img_clahe_rgb = cv2.cvtColor(imageInv, cv2.COLOR_GRAY2RGB)

        # 将 numpy 数组转换为 PIL 图像
        final_img = Image.fromarray(img_clahe_rgb)

        # img = Image.open(self.images_path[item])
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        

        label = self.images_class[item]

        if self.transform is not None:
            final_img = self.transform(final_img)

        return final_img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
