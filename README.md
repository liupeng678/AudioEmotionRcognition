# AudioEmotionRcognition
this is a repository about deep learning in emotion recognition





# Introduction 

## 1 项目概念



## 2 基础知识和论文思路

## 3 如何迭代项目

 ### 相关目录 



1. data中存放的样本，后续还有两个数据集会添加进来。
2. paper是写论文需要用到的图片和初稿。写论文在这里面去写。
3. related paper 是相关论文，包括自己借鉴的方法和有用的论文。这里面只放有用的。

 ###  代码

实验相关的代码都统一放到外面，这里面后续还要规范一下代码。代码的调整可以写到这: 



1. CompareExperimentOne.py 中存放了对比实验的代码和结果。这里面只需要修改模型就行。用于论文中RAVEE数据集的实验结果对比。



```
ResNet18, preprocess_input = Classifiers.get('seresnet34')
base_model = ResNet18(input_shape=(shape1,shape2,3), weights='imagenet', include_top=False)
```

这里面只需要跟换这些预训练模型的名字就行。如何更换参照：[here](https://github.com/qubvel/classification_models). 我们需要将里面能用到的预训练模型全部跑一遍。





实验结果添加到：待补充



1. OurModelOne.py 是我们自己设计模型，目前还在探索过程中，需要不断尝试才能确定最终的模型结构。













# Version log 

1. 添加基础的文件，搭建好了框架。