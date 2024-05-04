# 项目简介
本项目为上海交通大学 CS3511 课程 CodaLab Hypertensive-Retinopathy-Diagnosis-Challenge 的代码仓库。其中，主分支main为最后提交的代码版本迭代过程，SimpleCNN为使用简单CNN模型的代码版本，另外两个分支为使用EfficientNetV2过程中使用其他预处理方法的代码版本。具体可看报告。

# 项目结构
- `FinalCode`文件夹：存放了最终提交网站所需的文件。
- `Report`文件夹：存放了报告pdf和其LaTeX源码。
- `Sample`文件夹：存放了网站提供的样例代码。
- `Submit History`文件夹：存放了使用EfficientNetV2过程中所有版本的最终提交文件。
- `train-results`文件夹：存放了使用EfficientNetV2过程中所有版本的训练结果（包含训练结果输出和训练出来的模型参数）。
- `weights`文件夹：输出训练时的模型参数文件迭代。
- `data-raw.zip`文件：存放了网站提供的原始数据集。
- `draw-lines.py`，`transform_test.py`文件：参赛过程中使用的可视化文件。
- `model.py`文件：存放了EfficientNetV2模型的定义。
- `my_dataset.py`文件：存放了数据集的定义。
- `pre_efficientnetv2-s.pth`文件：存放了EfficientNetV2模型的初始参数。
- `predict.py`文件：存放了预测的代码，可供人工测试。
- `train.py`文件：存放了训练的代码。
- `utils.py`文件：存放了一些工具函数。

注1：EfficientNetV2模型的参考代码来源于[Github仓库：deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test11_efficientnetV2)

注2：本仓库未提供python环境，需自行配置。

注3：由于`Submit History`文件夹，`train-results`文件夹，`data-raw.zip`文件过大，故未上传至仓库，如需查看请联系我们。

# 使用说明
解压数据集，运行`train.py`文件即可训练模型。训练完成后，`weights`文件夹将输出训练时的模型参数文件迭代。
