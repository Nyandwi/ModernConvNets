# Modern Convolutional Neural Network Architectures

<p style='text-align: justify;'> <a href="https://nbviewer.jupyter.org/github/Nyandwi/convnets-architectures"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /> </a> <a href="https://colab.research.google.com/github/Nyandwi/convnets-architectures" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> </p>

***Revision of the designs and implementation of modern convolutional neural networks architectures***
-------
![cnns_image](images/gitcover.png)

Convolutional Neural Networks (ConvNets or CNNs) are a class of neural networks that are used for visual recognition tasks.

### ConvNets Architectures

* AlexNet - Deep Convolutional Neural Networks: [implementation](convnets/01-alexnet.ipynb), [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
* VGG - Very Deep Convolutional Networks for Large Scale Image Recognition: [implementation](convnets/02-vgg.ipynb), [paper](https://arxiv.org/pdf/1409.1556.pdf)
  
* GoogLeNet(Inceptionv1) - Going Deeper with Convolutions: [implementation](convnets/03-googlenet.ipynb), [paper](https://arxiv.org/abs/1409.4842)

* ResNet - Deep Residual Learning for Image Recognition: [implementation](convnets/04-resnet.ipynb), [annotated paper](annotated_papers/resnet.pdf) [paper](https://arxiv.org/abs/1512.03385)

* ResNeXt - Aggregated Residual Transformations for Deep Neural Networks: [implementation](convnets/06-resnext.ipynb), [annotated paper](annotated_papers/resnext.pdf), [paper](https://arxiv.org/abs/1611.05431v2)

* Xception - Deep Learning with Depthwise Separable Convolutions: [implementation](convnets/07-xception.ipynb), [annotated paper](annotated_papers/xception.pdf), [paper](https://arxiv.org/abs/1610.02357)

* DenseNet - Densely Connected Convolutional Neural Networks: [implementation](convnets/05-densenet.ipynb), [annotated paper](annotated_papers/densenet.pdf), [paper](https://arxiv.org/abs/1608.06993v5)

* MobileNetV1 - Efficient Convolutional Neural Networks for Mobile Vision Applications: [implementation](convnets/08-mobilenet.ipynb), [annotated paper](annotated_papers/mobilenet.pdf), [paper](https://arxiv.org/abs/1704.04861v1)

* MobileNetV2 - Inverted Residuals and Linear Bottlenecks: [implementation](convnets/09-mobilenetv2.ipynb) [annotated paper](annotated_papers/mobilenetv2.pdf), [paper](https://arxiv.org/abs/1801.04381)

* EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks: [implementation](convnets/10-efficientnet.ipynb), [annotated paper](annotated_papers/efficientnetv1.pdf), [paper](https://arxiv.org/abs/1905.11946v5). See also [EfficientNetV2](https://arxiv.org/abs/2104.00298v3)

* RegNet - Designing Network Design Spaces: [implementation](convnets/11-regnet.ipynb), [annotated paper](annotated_papers/regnet.pdf), [paper](hhttps://arxiv.org/abs/2003.13678). See also [this](https://arxiv.org/abs/2103.06877)

* ConvMixer - Patches are All You Need?: [implementation](convnets/12-convmixer.ipynb), [annotated paper](annotated_papers/convmixer.pdf), [paper](https://openreview.net/pdf?id=TVHS5Y4dNvM).

* ConvNeXt - A ConvNet for the 2020s: [implementation](convnets/13-convnext.ipynb), [annotated paper](annotated_papers/convnexts.pdf), [paper](https://arxiv.org/abs/2201.03545)

### On Choosing a ConvNets Architecture

Computer vision community is blessed with having many vision architectures that work great across many platforms or hardwares. But, having many options means it is not easy to choose an architecture that suits a given problem. How can you choose a CNNs architecture for your problem?

The first rule of thumb is that you should not try to design your own architecture from scratch. If you are working on generic problem, it never hurts to start with ResNet-50. If you are building a mobile-based visual application where there is limited computation resources, try MobileNets(or other mobile friendly architectures like [ShuffleNetv2](https://arxiv.org/abs/1807.11164) or [ESPNetv2](https://arxiv.org/abs/1811.11431)). 

For a better trade-off between accuracy and computation efficiency, I think [EfficientNetV2](https://arxiv.org/abs/2104.00298v3) and or latest [ConvNeXt](https://arxiv.org/abs/2201.03545) can be a good fit!

That said, choosing architecture is a no free-lunch scenario. There is not a going to be a single architecture that works for all datasets and problems. It's all experimentation. It's all trying!

If you are a visionary or like to stay on the bleeding edge of the field, try [vision transformers](https://paperswithcode.com/method/vision-transformer)!

### References Implementations

* [Keras Applications](https://github.com/keras-team/keras/tree/master/keras/applications)
* [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [PyTorch Vision](https://github.com/pytorch/vision)
* [Machine Learning Tokyo](https://github.com/Machine-Learning-Tokyo/CNN-Architectures)


### Important Notes

The implementations of ConvNets architectures contained in this repository are not optimized for training but rather to understand how those networks were designed, principal components that makes them and how they evolved overtime. LeNet-5(LeCunn, 1998) had 5 convolutional layers. AlexNet(Alex, 2012) had 9 convolutional layers. Few years later, Residual Networks(He, 2015) made the trends after showing that it's possible to train networks of over 100 layers. And in fact, residual networks are still one of the most widely used architecture across wide range of visual tasks and they impacted the [design of language architectures](https://arxiv.org/abs/2203.00555). Computer vision research community is very vibrant. Understanding how architectures are designed is not a neccesity, but it's one of the good ways to stay on top of this fast-ever changing field!

If you want to use ConvNets for solving a visual recognition tasks such as image classification or object detection, you can get up running quickly by getting the models (and their pretrained weights) from tools like [Keras](https://keras.io), [TensorFlow Hub](https://tfhub.dev), [PyTorch Vision](https://github.com/pytorch/vision), [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models), [GluonCV](https://cv.gluon.ai), and [OpenMML Lab](https://github.com/open-mmlab).

### Citation

If you find this repository helpful, I will appreciate if you cite it:

```
author: Jean de Dieu Nyandwi
title: ConvNets Architectures
year: 2022
publisher: GitHub
url: https://github.com/Nyandwi/convnets-architectures
```

For any suggestion, comment, or simply anything,you can reach out through [email]("mailto:johnjw7084@gmail.com), [Twitter](https://twitter.com/Jeande_d) or [LinkedIn](https://www.linkedin.com/in/nyandwi/).

***************************
![Twitter Follow](https://img.shields.io/twitter/follow/jeande_d?style=social)