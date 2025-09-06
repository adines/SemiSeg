# Semi-supervised learning methods for Semantic Segmentation


## Installation

Semi-Seg is available in PyPi for Python 3.10. To use it, you have to install 
Python 3.10 and pip.

```
pip install semisup-segmentation
```

## Architectures

We have studied the behaviour of 10 different semantic segmentation networks

|        Architecture        |Backbone | FLOPS (G)  | Parameters (M)           |
|----------------|-------------------------------|-----------------------------|----|
|CGNet| CGNet           |1.4 | 0.5 |
|DeepLabV3+          |ResNet50 | 14.8 | 28.7|
|DenseASPP  | ResNet50 | 14.1 | 29.1|
|FPENet  | FPENet | 0.3 | 0.1 |
|HRNet  | hrnet_w48 | 36.6 | 65.8 |
|LEDNet  | ResNet50 | 2.5 | 2.3 |
|MANet  | ResNet50 | 29.1 | 147.4 |
|OCNet  | ResNet50 | 16.9 | 35.9 |
|PAN  | ResNet50 | 13.6 | 24.3|
|U-Net  | ResNet50 | 48.5 | 13.4 |


## Distillation methods
We will compare three different methods:Data Distillation, Model Distillation and Data & Model Distillation; these methods are based on the notions of self-training and distillation.

### Data Distillation
In the case of Data Distillation, (1) a base model is trained, (2) this model is used to label new images using multiple transformations of the image, and (3) a new model is trained in both, the initial labelled images and the automatically annotated images in (2).

```
dataDistillation(baseModel, baseBackbone, targetModel, targetBackbone, transforms, path, outputPath, bs, size)
```

![workflow](assets/DataDistillation.svg)

**Parameters:**
+ baseModel (str): String with the name of the base segmentation architecture to be used to perform data distillation.
+ baseBackbone (str): String with the name of the base backbone architecture to be used to perform data distillation.
+ targetModel (str): String with the name of the segmentation architecture that will be trained at the end of the process.
+ targetBackbone (str): String with the name of the bacbone architecture that will be trained at the end of the process.
+ transforms (list\[str\]): List with the names of the different transformations to be applied to the images in the data distillation.
+ path (str): String with the path to the labelled images.
+ outputath (str): String with the path where the target model will be saved.
+ bs (int): Batch size of images used in training the models.
+ size ((int,int)): Image size used for model training.

### Model Distillation
In the case of Model Distillation (1) several models are trained in the initial annotated images, (2) these model are ensembled to label new images, and (3) a new model is trained in both, the initial labelled images and the automatically annotated images in (2).

```
modelDistillation(baseModels, baseBackbones, targetModel, targetBackbone, path, outputPath, bs, size)
```


![workflow](assets/ModelDistillation.svg)

**Parameters:**
+ baseModels (ist\[str\]): List with the names of the base segmentation architectures to be used to perform model distillation.
+ baseBackbones (list\[str\]): List with the name of the base backbone architectures to be used to perform model distillation.
+ targetModel (str): String with the name of the segmentation architecture that will be trained at the end of the process.
+ targetBackbone (str): String with the name of the bacbone architecture that will be trained at the end of the process.
+ path (str): String with the path to the labelled images.
+ outputath (str): String with the path where the target model will be saved.
+ bs (int): Batch size of images used in training the models.
+ size ((int,int)): Image size used for model training.


### Data & Model Distillation
Both techniques can also be combined in a technique called Data & Model Distillation.

```
modelDataDistillation(baseModels, baseBackbones, targetModel, targetBackbone, transforms, path, outputPath, bs, size)
```

![workflow](assets/DataModelDistillation.svg)

**Parameters:**
+ baseModels (ist\[str\]): List with the names of the base segmentation architectures to be used to perform model distillation.
+ baseBackbones (list\[str\]): List with the name of the base backbone architectures to be used to perform model distillation..
+ targetModel (str): String with the name of the segmentation architecture that will be trained at the end of the process.
+ targetBackbone (str): String with the name of the bacbone architecture that will be trained at the end of the process.
+ transforms (list\[str\]): List with the names of the different transformations to be applied to the images in the data distillation.
+ path (str): String with the path to the labelled images.
+ outputath (str): String with the path where the target model will be saved.
+ bs (int): Batch size of images used in training the models.
+ size ((int,int)): Image size used for model training.

## Datasets
In this work, we propose a benchmark of 3 biomedical datasets. 
For our study, we have split each of the datasets of the benchmark into two different sets: 
a training set with the $80 \%$ of images and a testing set with the $20\%$ of the images.

| Dataset                                                                                                                                    | Number of Images | Description                       |
|--------------------------------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------|
| [Kvasir-SEG](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/Ea3L9CLjb8xBsOga77t-puYB_1xL30SWeBGcshbdwJuOxQ?e=4c2aQ8) | 1000             | Gastrointestinal polyp images     |
| [COVID](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EQof7E22wGhHockzcQsdLEgBSHZbMVxLPPfyAex9zFvJ9A?e=SkctgU)      | 2724             | Lung CT scan dataset for COVID-19 |
| [BUSI](https://unirioja-my.sharepoint.com/:u:/g/personal/adines_unirioja_es/EVIrR3zz7JRGnxINME57FswBftH-TFFWV4lovA5t487_rA?e=Pvjc3x)       | 629              | Breast ultrasound images          |
