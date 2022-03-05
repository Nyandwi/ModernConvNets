# Modern Convolutional Neural Network Architectures
------
Revision of the designs and implementation of modern ConvNets architectures.
------
![cnns_image](images/gitcover.png)

Convolutional Neural Networks (ConvNets or CNNs) are classes of neural networks that are mostly used for visual recognition tasks.


### ConvNets Architectures

* AlexNet - Deep Convolutional Neural Networks: [implementation](convnets/01-alexnet.ipynb), [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
* VGG - Very Deep Convolutional Networks for Large Scale Image Recognition: [implementation](convnets/02-vgg.ipynb), [paper](https://arxiv.org/pdf/1409.1556.pdf)
  
* GoogLeNet(Inceptionv1) - Going Deeper with Convolutions: [implementation](convnets/03-googlenet.ipynb), [paper](https://arxiv.org/abs/1409.4842)

* ResNet - Deep Residual Learning for Image Recognition: [implementation](convnets/04-resnet.ipynb), [annotated paper](annotated_papers/resnet.pdf) [paper](https://arxiv.org/abs/1512.03385)

* ResNeXt - Aggregated Residual Transformations for Deep Neural Networks: [implementation](convnets/06-resnext.ipynb), [annotated paper](annotated_papers/resnext.pdf), [paper](https://arxiv.org/abs/1611.05431v2)

* Xception - Deep Learning with Depthwise Separable Convolutions: [implementation](convnets/07-xception.ipynb), [annotated paper](annotated_papers/xception.pdf), [paper](https://arxiv.org/abs/1610.02357)


* DenseNet - Densely Connected Convolutional Neural Networks: [implementation](convnets/05-densenet.ipynb), [annotated paper](annotated_papers/densenet.pdf), [paper](https://arxiv.org/abs/1608.06993v5)

* MobileNetV1 - Efficient Convolutional Neural Networks for Mobile Vision Applications: [implementation](convnets/08-mobilenet.ipynb), [annotated_paper](annotated_papers/mobilenet.pdf), [paper](https://arxiv.org/abs/1704.04861v1)

* MobileNetV2 - Inverted Residuals and Linear Bottlenecks: [implementation](convnets/09-mobilenetv2.ipynb) [annotated paper](annotated_papers/mobilenetv2.pdf), [paper](https://arxiv.org/abs/1801.04381)

* EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks: [implementation](convnets/10-efficientnet.ipynb), [annotated_paper](annotated_papers/efficientnetv1.pdf), [paper](https://arxiv.org/abs/1905.11946v5). See also [EfficientNetV2](https://arxiv.org/abs/2104.00298v3)

* RegNet - Designing Network Design Spaces: [implementation](convnets/11-regnet.ipynb), [annotated_paper](annotated_papers/regnet.pdf), [paper](hhttps://arxiv.org/abs/2003.13678). See also [this](https://arxiv.org/abs/2103.06877)

* ConvNeXt - A ConvNet for the 2020s: [implementation](convnets/10-convnext.ipynb), [annotated_paper](annotated_papers/convnexts.pdf), [paper](https://arxiv.org/abs/2201.03545)

* ConvMixer - Coming soon


For more about ConvNets, check out this introductory notebook.

### On Choosing a ConvNets Architecture



### References Implementations and Similar Repositories

* [Keras Applications](https://github.com/keras-team/keras/tree/master/keras/applications)
* [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [PyTorch Vision](https://github.com/pytorch/vision)
* [Machine Learning Tokyo](https://github.com/Machine-Learning-Tokyo/CNN-Architectures)


### Disclaimer

The implementations of ConvNets architectures contained in this repository are not optimized for training but rather to understand how those networks were designed, principal components that makes them and how they evolved overtime. LeNet-5(LeCunn, 1998) had 5 convolutional layers. AlexNet(Alex, 2012) had 9 convolutional layers. Few years later, Residual Networks(He, 2015) made the trends after showing that it's possible to train networks of over 100 layers. And in fact, residual networks are still one of the most widely used architecture across wide range of visual tasks and they impacted the design of other language architectures. 

If you want to use ConvNets for solving a visual recognition tasks such as image classification or object detection, you can get up running quickly by getting the models (and their pretrained weights) from tools like [Keras](https://keras.io), [TensorFlow Hub](https://tfhub.dev), [PyTorch Vision](https://github.com/pytorch/vision), [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models), [GluoCV](https://cv.gluon.ai), and [OpenMML Lab](https://github.com/open-mmlab).
