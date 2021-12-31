# Semantic-Segmentation-3D

## 一、简介

* 通过PyTorch框架实现的3D语义分割Demo，V-Net作为Baseline，采用Cross Entropy Loss和Dice Loss，评价指标包括Dice、Jaccard、95HD、ASD。
* 训练阶段采用Random Crop，测试阶段采用Sliding Window。
* 支持多GPU训练、单GPU测试，兼容单/多前景。
* 在实际任务中可以把本项目当作模版并修改或添加自己的代码。

## 二、说明

### 2.1 编写测试环境

* Python-3.7.11、PyTorch-1.4.0、Medpy-0.4.0

### 2.2 运行

* train.py文件，可通过如下命令进行多GPU训练```python train.py --cuda=2,3 --batch_size=8 --patch_size=128,128,96```，代表采用2、3号gpu，训练/测试阶段从原图裁剪patch_size大小的图片送入网络。
* test.py文件，默认使用单GPU，但由于评价指标均采用Medpy计算，无法通过GPU加速，故可修改代码使用CPU。
* process_data.py文件中可添加适合自己任务的数据处理代码并自定义保存格式，但相应的需要自行编写datasets文件夹下的自定义Dataset类，并修改train.py文件中相应代码。

### 2.3 参考链接

* <https://github.com/JunMa11/SegWithDistMap>
