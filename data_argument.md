# 数据增强技术

## 常用方法
* Color Jittering：对颜色的数据增强：图像亮度、饱和度、对比度变化（此处对色彩抖动的理解不知是否得当）
* PCA  Jittering：首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，
得到特征向量和特征值，用来做PCA Jittering；
* Random Scale：尺度变换；
* Random Crop：采用随机图像差值方式，对图像进行裁剪、缩放；
包括Scale Jittering方法（VGG及ResNet模型使用）或者尺度和长宽比增强变换；
* Horizontal/Vertical Flip：水平/垂直翻转；
* Shift：平移变换
* Rotation/Reflection：旋转/仿射变换；
* Noise：高斯噪声、模糊处理；
* Label shuffle：类别不平衡数据的增广，参见海康威视ILSVRC2016的report；
另外，文中提出了一种Supervised Data Augmentation方法，有兴趣的朋友的可以动手实验下。
    
    
    

## 一些论文方法

### Some Improvements on Deep Convolutional Neural Network Based Image Classification
* [paper](paper/Some%20Improvements%20on%20Deep%20Convolutional%20Neural%20Network%20Based%20Image%20Classification.pdf)
    * GAN生成数据
    * 利用网络训练增强

### AutoAugment
* [谷歌把数据增强也自动化了，ImageNet数据集准确率创新高！](https://mp.weixin.qq.com/s/8cjPs0cvzgd60CXQVGja0A)
    * [paper: AutoAugment](paper/2018%20-%20AutoAugment%20Learning%20Augmentation%20Policies%20from%20Data.pdf)
    * [github code](data_argument/AutoAugment.py)
    
    
### Cutout方法   
* [paper](paper/2017-Improved%20Regularization%20of%20Convolutional%20Neural%20Networks%20with%20Cutout.pdf)
* 通过加入掩码的方式增强数据: 将输入图片进行部分遮挡，鼓励模型不依赖那些特殊区域表示。
* [github](data_argument/cutout.py)
* for cifar10 cutout=16


### Pairing Samples 方法
* [paper](paper/2018-Data%20Augmentation%20by%20Pairing%20Samples%20for%20Images%20Classification.pdf)
* 训练的时候，通过随机选择另一张图像，然后mix两张图像的方式，产生一张新的图像，这张图像的标签是第一张图像的标签，
* 通过这样的方式增大了数据量，然后训练网络，
* 等到网络稳定后，再使用原来的图像和标签，进行微调，得到最终的模型。


### The Effectiveness of Data Augmentation in Image Classification using Deep Learning
* [paper](paper/2017-The%20Effectiveness%20of%20Data%20Augmentation%20in%20Image%20Classification%20using%20Deep%20Learning.pdf)
* 


    
* [数据增强：数据有限时如何使用深度学习](https://www.leiphone.com/news/201805/avOH5g1ZX3lAbmjp.html) 
    * 当你训练一个机器学习模型时，你实际做工作的是调参，以便将特定的输入（一副图像）映像到输出（标签）。
    我们优化的目标是使模型的损失最小化， 以正确的方式调节优化参数即可实现这一目标。
    * 因为这是大多数机器学习算法就是这么工作的。它会寻找区分一个类和另一个类的最明显特征。
    在这个例子中 ，这个特征就是所有品牌A的汽车朝向左边，所有品牌B的汽车朝向右边。