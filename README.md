# ConvNets Architectures
------
Revision of the designs and implementation of modern Convolutional Neural Networks.
------
![cnns_image](images/gitcover.png)

Convolutional Neural Networks (a.k.a ConvNets or CNNs) are classes of neural networks that are mostly used for image recognition tasks.


### Covered ConvNets Architectures

* AlexNet - Deep Convolutional Neural Networks: [implementation](convnets/1-alexnet.ipynb), [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
* VGG - Very Deep Convolutional Networks for Large Scale Image Recognition: [implementation](convnets/2-vgg.ipynb), [paper](https://arxiv.org/pdf/1409.1556.pdf)
  
* GoogLeNet(Inceptionv1) - Going Deeper with Convolutions: [implementation](convnets/3-googlenet.ipynb), [paper](https://arxiv.org/abs/1409.4842)

* ResNet - Deep Residual Learning for Image Recognition: [implementation](convnets/4-resnet.ipynb), [annotated paper](annotated_papers/resnet.pdf) [paper](https://arxiv.org/abs/1512.03385)

* ResNeXt - Aggregated Residual Transformations for Deep Neural Networks: [implementation](convnets/6-resnext.ipynb), [annotated paper](annotated_papers/resnext.pdf), [paper](https://arxiv.org/abs/1611.05431v2)

* Xception - Deep Learning with Depthwise Separable Convolutions: [implementation](convnets/7-xception.ipynb), [annotated paper](annotated_papers/xception.pdf), [paper](https://arxiv.org/abs/1610.02357)


* DenseNet - Densely Connected Convolutional Neural Networks: [implementation](convnets/5-densenet.ipynb), [annotated paper](annotated_papers/densenet.pdf), [paper](https://arxiv.org/abs/1608.06993v5)

* MobileNetV1 - Efficient Convolutional Neural Networks for Mobile Vision Applications: [implementation](convnets/8_mobilenet.ipynb), [annotated_paper](annotated_papers/mobilenet.pdf), [paper](https://arxiv.org/abs/1704.04861v1)

* MobileNetV2 - Inverted Residuals and Linear Bottlenecks: [implementation](convnets/9-mobilenetv2.ipynb) [annotated paper](annotated_papers/mobilenetv2.pdf), [paper](https://arxiv.org/abs/1801.04381)

* EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks: [implementation](convnets/10-efficientnet.ipynb), [annotated_paper](annotated_papers/efficientnetv1.pdf), [paper](https://arxiv.org/abs/1905.11946v5). See also [EfficientNetV2](https://arxiv.org/abs/2104.00298v3)

* ConvNeXt - A ConvNet for the 2020s: [implementation](convnets/10-convnext.ipynb), [annotated_paper](annotated_papers/convnexts.pdf), [paper](https://arxiv.org/abs/2201.03545)

* RegNetY - Coming soon
* ConvMixer - Coming soon


For more about ConvNets, check out this introductory notebook.


### References Implementations and Similar Repositories

* Keras Applications
* Timm
* PyTorch Vision
* ML Tokyo


### Disclaimer

The implementations of ConvNets architectures contained in this repository are not optimized for training but rather to understand how those networks were designed, principal components that makes them and how they evolved overtime. LeNet-5(LeCunn, 1998) had 5 convolutional layers. AlexNet(Alex, 2012) had 9 convolutional layers. Few years later, Residual Networks(He, 2015) made the trends after showing that it's possible to train networks of over 100 layers. And in fact, residual networks are still one of the most widely used architecture across wide range of visual tasks and it impacted the design of other language architectures. Currently, there are lots going on such as visual attentions.

If you want to use ConvNets for solving a visual recognition tasks such as image classification or object detection, you can get up running quickly by getting the models (and their pretrained weights) from tools like Keras, TensorFlow Hub, PyTorch Vision, Timm, GluoCV, and OpenMML Lab.
