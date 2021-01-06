# 基于心电图分析的心脏疾病诊断

本项目是对来自上海某医院的真实数据的分析，题目描述见kaggle：[传送门](https://www.kaggle.com/c/statlearning-sjtu-2020/overview)，这里给出代码和运行方法。本项目的组织稍显冗余，实际上项目中多种传统机器学习方法的程序结构相似，可以使用多态的方法合并起来，将来如果有机会再重构代码。

## 准备工作

本项目在MacOS Big Sur 11.1下开发，对Linux和Windows系统应具有兼容性。首先配置Python虚拟环境。

```bash
conda create -n python38 python==3.8.3
```

安装相关依赖环境，这里可能没有罗列完整。如果在运行过程中出现缺失，请使用pip或conda等工具安装

```bash
pip install matplotlib, biosppy, scipy, pywt, numpy, sklearn
conda install tensorflow=2.4
```

本项目的目录结构大体如下。由于Python的包管理有一些bug，本项目使用`Python -m workspace...`的形式运行程序，并保持当前路径为workspace的上一级。最终生成的测试结果存储在`submission.csv`中。

```
workspace
├── ECGTestData
├── ECGTrainData
├── data
├── Preprocess
│     ├── christov_643.txt
│     ├── hamilton_19.txt
│     ├── naive_7.txt
│     ├── process_data.py
│     └── r_peak_evaluate.py
├── CNN
├── KNN
├── SVM
├── DT
├── ICA
└── submission.csv
```

配置kaggle环境，并使用以下命令下载数据集

```bash
kaggle competitions download -c statlearning-sjtu-2020
```

之后解压文件，将相关数据存放在项目目录的`ECGTestData`和`ECGTrainData`文件夹下，保持文件夹层级关系不变。运行数据预处理命令

```bash
python -m workspace.Preprocess.process_data
```

这时在`data`文件夹下将会生成5折交叉校验的训练集、验证集，以及最终的测试集。这些数据是将原始数据拼接后进行了R峰检测与分割、db8小波降噪和shuffle后得到的标准数据集，存储格式为Numpy的二进制文件格式`*.npy`。运行以下命令可以得到不同的ECG的R峰检测算法的性能对比。

```bash
python -m workspace.Preprocess.r_peak_evaluate
```

## 卷积神经网络

训练神经网络得到交叉校验的准确率
```bash
python -m workspace.CNN.train
```
模型将会被储存在 `workspace/CNN/model/`，日志文件将会被储存在`workspace/CNN/logs/`。可以通过TensorBoard监控模型的训练过程，Mac与Linux上后台执行该功能的命令为
```bash
nohup tensorboard --log_dir=workspace/CNN/logs >/dev/null 2>&1 &
```

可以通过如下命令查看不同神经网络结构的性能曲线。程序中的数据，是通过交叉验证得到的准确率。

```bash
python -m workspace.CNN.plot
```

最后进行测试得到最终待提交的结果

```bash
python -m workspace.CNN.test
```

## 支持向量机

训练模型，得到交叉校验的准确率

```bash
python -m workspace.SVM.svm
```

可以通过如下命令查看模型选择的性能曲线

```bash
python -m workspace.SVM.plot
```

最后进行测试得到最终待提交的结果

```bash
python -m workspace.SVM.svm --test
```

## K-近邻

训练模型，得到交叉校验的准确率

```bash
python -m workspace.KNN.knn
```

可以通过如下命令查看模型选择的性能曲线

```bash
python -m workspace.KNN.plot
```

最后进行测试得到最终待提交的结果

```bash
python -m workspace.KNN.knn --test
```

## 决策树

训练模型，得到交叉校验的准确率

```bash
python -m workspace.DT.dt
```

进行测试得到最终待提交的结果

```bash
python -m workspace.DT.dt --test
```

## 独立成分分析

以上除卷积神经网络使用原始数据之外，默认使用原始数据降采样，与快速傅里叶变换得到的频域信息插值结果，合并作为输入的特征。这里使用独立成分分析的方式，使用8个分量的权重向量，结合上述三种传统方法重新进行实验。

训练模型，得到交叉校验的准确率

```bash
python -m workspace.ICA.ica
```

选择对不同的模型进行测试，`method`后面的参数可以是`knn`，`svm`和`dt`

```bash
python -m workspace.ICA.ica --test --method knn
```