# NetWork
## YOLO 上手 --ultralytics

概念详解:

cuda、nvcc、GPU、深度学习框架、N卡驱动、pytorch、cudnn、TensorRT、onnx、anaconda、python、cuda Toolkit...


## [从全连接层开始](https://www.bilibili.com/video/BV1hE411t7RN)

### nn.Conv2d 参数详解:
[卷积动画制作](https://github.com/vdumoulin/conv_arithmetic)

dilation:

[​​dilation动画详解](https://blog.csdn.net/weixin_42363544/article/details/123920699)

groups:

0.[1x1卷积核的作用](https://zhuanlan.zhihu.com/p/40050371):理解为全连接层

1.[深度可分离卷积](https://blog.csdn.net/m0_37605642/article/details/134174749)

2.传统卷积和深度可分离卷积对比:后者使用了每个通道的1张的卷积1次后的特征图来进行线性组合，实现特征升维；

  而传统卷积是对输入的每个通道进行不同的卷积核处理，直接分发出不同的特征图，进行简单加和，本身的特征维度就很高了；

3.怎么结合二者，对输入的每个通道进行不同卷积核的处理，提取出第1次卷积后的特征图，每个通道均有自己的不同的特征图，再对此进行1x1卷积核的线性组合，实现丰富特征的线性组合。

4.分析传统卷积和深度可分离卷积的计算量，通过实验证明，利用线性组合的方式弥补牺牲的一定数量的特征，效果依旧尚佳，但是大大降低计算量，加快训练和推理的速度

Tips:如何权衡牺牲性能来降低计算量:重点是部署后推理的速度与准确性；

## [从minst重新看神经网络](https://github.com/Functionhx/NetWork/blob/master/0minst.py)

## [魔改YOLO](https://blog.csdn.net/m0_67647321/article/details/143481224"点击访问付费专栏")
