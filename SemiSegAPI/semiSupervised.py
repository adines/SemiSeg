from SemiSegAPI.utils import *
from fastai.vision.all import *
import shutil
import os
import gc


def dataDistillation(baseModel, baseBackbone, targetModel, targetBackbone, transforms, path, outputPath, bs=32,
                     size=(480, 640)):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    elif not testTransforms(transforms):
        print("There are invalid transforms")
    else:
        # Load data and model
        dls = get_dls(path, size, bs=bs)
        nClasses = getNumClasses(path)
        learn = getLearner(baseModel, baseBackbone, nClasses, path, dls)

        # Train base learner
        print('Training ' + baseModel + ' model')
        train_learner(learn, 20, freeze_epochs=2)

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbone + '.pth',
                    outputPath + os.sep + 'base_' + baseModel + '_' + baseBackbone + '.pth')

        # Free GPU memory
        del learn
        del dls
        gc.collect()
        torch.cuda.empty_cache()

        # supervised method
        print("Start of annotation")
        omniData(path, baseModel, baseBackbone, transforms, size)
        print("End of annotation")

        # Load new images
        dls2 = get_dls(path + '_tmp', size, bs=bs)

        # Load base model
        learn2 = getLearner(targetModel, targetBackbone, nClasses, path + '_tmp', dls2)

        # Train target learner
        print('Training ' + targetModel + ' model')
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')

        # Free GPU memory
        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()


def modelDistillation(baseModels, baseBackbones, targetModel, targetBackbone, path, outputPath, bs=32, size=(480, 640)):
    for baseModel in baseModels:
        if not testNameModel(baseModel):
            print("The base model selected is not valid")
            return
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        nClasses = getNumClasses(path)

        for baseModel, baseBackbone in zip(baseModels, baseBackbones):
            # Load data and model
            dls = get_dls(path, size, bs=bs)
            learn = getLearner(baseModel, baseBackbone, nClasses, path, dls)

            # Train base learner
            print('Training ' + baseModel + ' model')
            train_learner(learn, 20, freeze_epochs=2)
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbone + '.pth',
                        outputPath + os.sep + 'base_' + baseModel + '_' + baseBackbone + '.pth')

            # Free GPU memory
            del learn
            del dls
            gc.collect()
            torch.cuda.empty_cache()

        # supervised method
        print("Start of annotation")
        omniModel(path, baseModels, baseBackbones, size)
        print("End of annotation")

        # Load new images
        dls2 = get_dls(path + '_tmp', size, bs=bs)

        # Load base model
        learn2 = getLearner(targetModel, targetBackbone, nClasses, path + '_tmp', dls2)

        # Train target learner
        print('Training ' + targetModel + ' model')
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')

        # Free GPU memory
        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()


def modelDataDistillation(baseModels, baseBackbones, targetModel, targetBackbone, transforms, path, outputPath, bs=32,
                          size=(480, 640)):
    for baseModel in baseModels:
        if not testNameModel(baseModel):
            print("The base model selected is not valid")
            return
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    elif not testTransforms(transforms):
        print("There are invalid transforms")
    else:
        nClasses = getNumClasses(path)

        for baseModel, baseBackbone in zip(baseModels, baseBackbones):
            # Load data and model
            dls = get_dls(path, size, bs=bs)
            learn = getLearner(baseModel, baseBackbone, nClasses, path, dls)

            # Train base model
            print('Training ' + baseModel + ' model')
            train_learner(learn, 20, freeze_epochs=2)
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbone + '.pth',
                        outputPath + os.sep + 'base_' + baseModel + '_' + baseBackbone + '.pth')
            # Free GPU memory
            del learn
            del dls
            gc.collect()
            torch.cuda.empty_cache()

        # supervised method
        print("Start of annotation")
        omniModelData(path, baseModels, baseBackbones, transforms, size)
        print("End of annotation")

        # Load new images
        dls2 = get_dls(path + '_tmp', size, bs=bs)

        # Load target model
        learn2 = getLearner(targetModel, targetBackbone, nClasses, path + '_tmp', dls2)

        # Train target learner
        print('Training ' + targetModel + ' model')
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')

        # Free GPU memory
        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()


def simpleTraining(baseModel, baseBackbone, path, outputPath, bs=32, size=(480, 640)):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        # Load data and model
        dls = get_dls(path, size, bs=bs)
        nClasses = getNumClasses(path)
        learn = getLearner(baseModel, baseBackbone, nClasses, path, dls)

        # Train base model
        print('Training ' + baseModel + ' model')
        train_learner(learn, 20, freeze_epochs=2)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbone + '.pth',
                    outputPath + os.sep + 'target_' + baseModel + '_' + baseBackbone + '.pth')

        # Free GPU memory
        del learn
        del dls
        gc.collect()
        torch.cuda.empty_cache()