import os
import torch
import torch.nn as nn
import torch.optim as optim
from network import CNN, preprocess_image
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定图片所在文件夹路径
folder_path = "data/full_data"

# 获取文件夹中所有图片文件的文件名，并按文件名排序
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

dataset = []
for image_file in image_files:
    # 构建图片文件的完整路径
    image_path = os.path.join(folder_path, image_file)
    
    # 使用OpenCV读取图片
    image = cv2.imread(image_path)
    image_tensor = preprocess_image(image)
    dataset.append(image_tensor)

dataset = torch.stack(dataset).to(device)  # 将数据移到 GPU 上

# 读取标签文件
labels_df = pd.read_csv('data/Labels.csv', header=None, names=['filename', 'label'], dtype={'filename': str, 'label': str})
labels = []
index_0 = []
index_1 = []
for i, row in labels_df.iterrows():
    if i == 0:
        continue
    if row['label'] == '1':
        label = 1
        index_1.append(i-1)
    else:
        label = 0
        index_0.append(i-1)
    labels.append(label)

# 保证标签为 0 和标签为 1 的样本数量相同
min_count = min(len(index_0), len(index_1))
index_0 = random.sample(index_0, min_count)
index_1 = random.sample(index_1, min_count)
index = index_0 + index_1
sampled_dataset = [dataset[i] for i in index]
sampled_labels = [labels[i] for i in index]

sampled_dataset = torch.stack(sampled_dataset).to(device)
sampled_labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)

# 划分训练集和测试集
train_files, test_files, train_labels, test_labels = train_test_split(sampled_dataset, sampled_labels, test_size=0.3, random_state=42)
count_test_0 = 0
count_test_1 = 0
for i in test_labels:
    if i == 0:
        count_test_0 += 1
    else:
        count_test_1 += 1

print('Downloading finished.')

# 创建模型实例并将其移到 GPU 上
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置批量大小
batch_size = 32

# 训练模型
num_epochs = 50
per_epoch = 2
total_batches = len(train_files) // batch_size
accuracies = []
max_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(total_batches):
        batch_images = train_files[i*batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
        batch_labels = torch.tensor(batch_labels, dtype=torch.float).unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_images.size(0)
    epoch_loss = running_loss / len(train_files)
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 每5个 epoch 结束后评估模型在测试集上的性能
    if (epoch + 1) % per_epoch == 0:
        model.eval()
        correct = 0
        total = 0
        recall_0 = 0
        recall_1 = 0
        with torch.no_grad():
            for images, labels in zip(test_files, test_labels):
                images = images.unsqueeze(0)  # 添加一个 batch_size 维度
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.float).unsqueeze(0).to(device)
                outputs = model(images)
                #predicted = torch.round(outputs)
                predicted = 1 if outputs.item() >= 0.5 else 0
                #predicted_int = int(predicted.item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                recall_0 += (predicted == labels).sum().item() if labels == 0 else 0
                recall_1 += (predicted == labels).sum().item() if labels == 1 else 0
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f'Epoch: {epoch+1} Test Accuracy: {accuracy:.4f} Recall 0: {recall_0}/{count_test_0} Recall 1: {recall_1}/{count_test_1}')
        '''print(f'Epoch: {epoch+1}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Recall 0: {recall_0}/{count_test_0}')
        print(f'Recall 1: {recall_1}/{count_test_1}')'''
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), 'model.pth')


# 保存模型
#torch.save(model.state_dict(), 'model.pth')

# 制作图像
plt.plot(range(1, num_epochs+1, per_epoch), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs. Epoch')
plt.show()