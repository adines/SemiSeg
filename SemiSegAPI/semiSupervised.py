from SemiSegAPI.utils import *
from fastai.vision.all import *
import shutil
import os
import gc

def dataDistillation(baseModel, baseBackbone, targetModel, targetBackbone, transforms, path, outputPath, bs=32, size=(480,640)):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    elif not testTransforms(transforms):
        print("There are invalid transforms")
    else:
        # Load images
        dls = get_dls(path, size, bs=bs)
        nClasses=numClasses(path)

        learn = getLearner(baseModel,baseBackbone,nClasses,path,dls)

        # Train base learner
        print("Start of base model training")
        print('Training '+ baseModel+ ' model')
        train_learner(learn, 20, freeze_epochs=2)

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path + os.sep + 'models' + os.sep + baseModel+'_'+baseBackbone + '.pth',
                    outputPath + os.sep + 'base_' + baseModel+'_'+baseBackbone + '.pth')

        del learn
        del dls
        gc.collect()
        torch.cuda.empty_cache()

        # supervised method
        print("Start of annotation")
        omniData(path, baseModel,baseBackbone, transforms,size)
        print("End of annotation")



        # Load new images
        dls2 = get_dls(path + '_tmp', size, bs=bs)

        # Load base model
        learn2 = getLearner(targetModel, targetBackbone, nClasses, path + '_tmp', dls2)

        # Train base learner
        print("Start of target model training")
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')

        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()


def modelDistillation(baseModels, baseBackbones, targetModel, targetBackbone, path, outputPath, bs=32, size=(480,640)):
    for baseModel in baseModels:
        if not testNameModel(baseModel):
            print("The base model selected is not valid")
            return
    if not testNameModel(targetModel):
        print("The target model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        # Load images

        nClasses = numClasses(path)



        # Load base model
        print("Start of base models training")
        for i,baseModel in enumerate(baseModels):
            dls = get_dls(path, size, bs=bs)
            learn = getLearner(baseModel, baseBackbones[i], nClasses, path, dls)


            # Train base learner
            train_learner(learn, 5, freeze_epochs=2)
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbones[i] + '.pth',
                        outputPath + os.sep + 'base_' + baseModel + '_' + baseBackbones[i] + '.pth')

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

        # Train base learner
        print("Start of target model training")
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')
        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()


def modelDataDistillation(baseModels, baseBackbones, targetModel, targetBackbone, transforms, path, outputPath, bs=32, size=(480,640)):
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
        nClasses = numClasses(path)

        # Load images
        print("Start of base models training")
        for i,baseModel in enumerate(baseModels):
            dls = get_dls(path, size, bs=bs)
            learn = getLearner(baseModel, baseBackbones[i], nClasses, path, dls)

            # Train base learner
            train_learner(learn, 5, freeze_epochs=2)
            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            shutil.copy(path + os.sep + 'models' + os.sep + baseModel + '_' + baseBackbones[i] + '.pth',
                        outputPath + os.sep + 'base_' + baseModel + '_' + baseBackbones[i] + '.pth')
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

        # Load base model
        learn2 = getLearner(targetModel, targetBackbone, nClasses, path + '_tmp', dls2)

        # Train base learner
        print("Start of target model training")
        train_learner(learn2, 20, freeze_epochs=2)
        shutil.copy(path + '_tmp' + os.sep + 'models' + os.sep + targetModel + '_' + targetBackbone + '.pth',
                    outputPath + os.sep + 'target_' + targetModel + '_' + targetBackbone + '.pth')
        shutil.rmtree(path + '_tmp')
        del learn2
        del dls2
        gc.collect()
        torch.cuda.empty_cache()

def simpleTraining(baseModel, baseBackbone, path, outputPath, bs=32, size=(480,640)):
    if not testNameModel(baseModel):
        print("The base model selected is not valid")
    elif not testPath(path):
        print("The path is invalid or has an invalid structure")
    else:
        # Load images
        dls = get_dls(path, size, bs=bs)
        nClasses = numClasses(path)
        learn = getLearner(baseModel, baseBackbone, nClasses, path, dls)

        # Train base learner
        print("Start of model training")
        train_learner(learn, 20, freeze_epochs=2)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        shutil.copy(path+os.sep+'models'+os.sep+baseModel+'_'+baseBackbone+'.pth',outputPath+os.sep+'target_'+baseModel+'_'+baseBackbone+'.pth')