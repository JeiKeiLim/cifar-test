# cifar-test
Cifar10/100 Training, Pruning, Quantization Test


# train_base_model_cifar.py
## Example
### 1. Training Base Model
#### 1.1. Training From Scratch
```shell
python train_base_model_cifar.py --model resnet50 --dataset cifar10 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```

#### 1.2. Resume Training
```shell
python train_base_model_cifar.py --model resnet50 --weights ./export/saved_weight.h5 --dataset cifar10 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```
#### 1.3. Fine Tuning
```shell
python train_base_model_cifar.py --model resnet50 --weights ./export/saved_weight.h5 --dataset cifar10 --epochs 100 --unfreeze 10 --lr 0.0001 --img_w 32 --img_h 32
```

### 2. Distillation
#### 2.1. Distill from teacher model
```shell
python train_base_model_cifar.py --model resnet18 --distill --teacher ./export/saved_model.h5 --temperature 2.0 --dataset cifar10 --epochs 100 --unfreeze -1 --lr 0.001 --img_w 32 --img_h 32
```



## Usage
```
usage: train_base_model_cifar.py [-h] [--model MODEL] [--dataset DATASET]
                                 [--unfreeze UNFREEZE] [--lr LR]
                                 [--batch BATCH] [--epochs EPOCHS]
                                 [--weights WEIGHTS] [--summary]
                                 [--img_w IMG_W] [--img_h IMG_H] [--distill]
                                 [--teacher TEACHER]
                                 [--temperature TEMPERATURE]
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model Name. Supported Models: (vgg16, vgg19,
                        mobilenetv2, mobilenetv1, effnetb0, effnetb1,
                        effnetb2, effnetb3, effnetb4, effnetb5, effnetb6,
                        effnetb7, effnetl2, resnet101, resnet101v2, resnet50,
                        resnet50v2, resnet152, resnet152v2, inceptionv3,
                        inceptionresnetv2, densenet121, densenet169,
                        densenet201, nasnetlarge, nasnetmobile, xception,
                        resnet18, resnet10). (Default: resnet18)
  --dataset DATASET     Dataset Name. Supported Dataset: (cifar10, cifar100).
                        (Default: cifar10)
  --unfreeze UNFREEZE   A number unfreeze layer. 0: Freeze all. -1: Unfreeze
                        all. (Default: 0)
  --lr LR               Learning Rate. (Default: 0.001)
  --batch BATCH         Batch Size. (Default: 8)
  --epochs EPOCHS       Epochs. (Default: 1000)
  --weights WEIGHTS     Weight path to load. If not given, training begins
                        from scratch with imagenet base weights
  --summary             Display a summary of the model and exit
  --img_w IMG_W         Image Width (Default: 224)
  --img_h IMG_H         Image Height (Default: 224)
  --distill             Perform Distillation
  --teacher TEACHER     Teacher Model Path
  --temperature TEMPERATURE
                        Soft Label Temperature

```

