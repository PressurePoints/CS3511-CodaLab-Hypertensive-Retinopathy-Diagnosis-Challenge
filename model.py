import os
import cv2
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image



class model:
    def __init__(self):
        self.checkpoint = "model.pth"
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.model = CNN()
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) 
        image = preprocess_image(image_rgb)

        with torch.no_grad():
            image = image.unsqueeze(0)  # 添加一个 batch_size 维度
            outputs = self.model(image)
            predicted = 1 if outputs.item() > 0.5 else 0

        return predicted


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

    # 对绿色通道进行对比度有限的自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imageEqualized = clahe.apply(imageGreen)
    
    imageEqualized_r = clahe.apply(r)
    imageEqualized_b = clahe.apply(b)
    imageEqualized_rgb = cv2.merge([imageEqualized_r, imageEqualized, imageEqualized_b])

    img_pil = Image.fromarray(imageEqualized_rgb)
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img_pil)
