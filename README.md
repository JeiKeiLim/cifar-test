# cifar-test
* Tensorflow cifar10 and cifar100 training experiments

# Support functions
1. Pruning
2. Distillation
3. Self-Distillation
4. Quantization
5. Ensemble
6. Test time augmentation
7. Multi-GPU training

# Requirements
* tfhelper >= 0.0.41
* efficientnet>=1.1.0

# train_base_model_cifar.py
## Example
### 1. Training Base Model
#### 1.1. Training From Scratch
```shell
python train_base_model_cifar.py --model resnet50 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```

#### 1.2. Resume Training
```shell
python train_base_model_cifar.py --model resnet50 --weights ./export/saved_weight.h5 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```
#### 1.3. Fine Tuning
```shell
python train_base_model_cifar.py --model resnet50 --weights ./export/saved_weight.h5 --epochs 100 --unfreeze 10 --lr 0.0001 --img_w 32 --img_h 32
```

#### 1.4. Augmentation
```shell
python train_base_model_cifar.py --model resnet18 --epochs 100 --img_w 32 --img_h 32 --augment auto --augment-policy imagenet
```

### 2. Distillation
#### 2.1. Distill from teacher model
```shell
python train_base_model_cifar.py --model resnet18 --distill --teacher ./export/saved_model.h5 --temperature 2.0 --epochs 100 --unfreeze -1 --lr 0.001 --img_w 32 --img_h 32
```

### 3. Self-Distillation
```shell
python train_base_model_cifar.py --model resnet18 --self-distill --temperature 2.0 --epochs 100 --unfreeze -1 --lr 0.001 --img_w 32 --img_h 32
```

### 4. Custom Model Architecture
#### 4.1. ResNet18, ResNet10
ResNet uses first channel size as 64. You can adjust this size as below.
```shell
python train_base_model_aihub.py --model resnet10 --resnet-init-channel 32 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```

#### 4.2. MicroJKNet
DenseNet based Architecture. You can customize depth and width of the model.
- growth-rate: Number of channel stacked by model-in-depth
- model-depth: Number of Dense block.
- model-in-depth: Number of layers within Dense block.

```shell
python train_base_model_aihub.py --model microjknet --growth-rate 12 --model-depth 3 --model-in-depth 3 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```

#### 4.3. Training with float16
```shell
python train_base_model_aihub.py --float16 --float16-dtype mixed_float16 --model microjknet --growth-rate 12 --model-depth 3 --model-in-depth 3 --epochs 100 --lr 0.001 --img_w 32 --img_h 32
```


## Usage
```
usage: train_base_model_cifar.py [-h] [--model MODEL] [--dataset DATASET]
                                 [--unfreeze UNFREEZE] [--top-layer TOP_LAYER]
                                 [--lr LR] [--batch BATCH] [--epochs EPOCHS]
                                 [--weights WEIGHTS] [--weights-no-build]
                                 [--summary] [--img_w IMG_W] [--img_h IMG_H]
                                 [--resnet-init-channel RESNET_INIT_CHANNEL]
                                 [--distill] [--teacher TEACHER]
                                 [--skip-teacher-eval]
                                 [--temperature TEMPERATURE] [--self-distill]
                                 [--no-tensorboard] [--no-tensorboard-writing]
                                 [--tboard-root TBOARD_ROOT]
                                 [--tboard-host TBOARD_HOST]
                                 [--tboard-port TBOARD_PORT]
                                 [--tboard-profile TBOARD_PROFILE]
                                 [--tboard-update-freq TBOARD_UPDATE_FREQ]
                                 [--logistic-hidden-ratio LOGISTIC_HIDDEN_RATIO]
                                 [--debug] [--float16]
                                 [--float16-dtype FLOAT16_DTYPE]
                                 [--growth-rate GROWTH_RATE]
                                 [--model-depth MODEL_DEPTH]
                                 [--model-in-depth MODEL_IN_DEPTH]
                                 [--compression-rate COMPRESSION_RATE]
                                 [--expansion EXPANSION] [--augment AUGMENT]
                                 [--augment-policy AUGMENT_POLICY]
                                 [--augment-test] [--activation ACTIVATION]
                                 [--dropout DROPOUT] [--conv CONV]
                                 [--data-format DATA_FORMAT] [--seed SEED]
                                 [--prefetch] [--test-only]
                                 [-en [ENSEMBLE_MODELS [ENSEMBLE_MODELS ...]]]
                                 [--multi-gpu] [--multi-worker MULTI_WORKER]
                                 [--save-metric SAVE_METRIC]
                                 [--metric-type METRIC_TYPE] [--tta]
                                 [--n-tta N_TTA] [--tta-softmax TTA_SOFTMAX]
                                 [--no-tta-softmax] [--tta-mp]
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model Name. Supported Models: (vgg16, vgg19,
                        mobilenetv2, mobilenetv1, effnetb0, effnetb1,
                        effnetb2, effnetb3, effnetb4, effnetb5, effnetb6,
                        effnetb7, effnetl2, resnet101, resnet101v2, resnet50,
                        resnet50v2, resnet152, resnet152v2, inceptionv3,
                        inceptionresnetv2, densenet121, densenet169,
                        densenet201, nasnetlarge, nasnetmobile, xception,
                        resnet18, resnet10, microjknet, logistic, ensemble).
                        (default: resnet18)
  --dataset DATASET     Dataset Name. Supported Dataset: (cifar10, cifar100).
                        (default: cifar10)
  --unfreeze UNFREEZE   A number unfreeze layer. 0: Freeze all. -1: Unfreeze
                        all. (default: 0)
  --top-layer TOP_LAYER
                        Top Layer Type(gap: 1x1conv->GAP, dense:
                        Flatten->Fully Connected (default: gap)
  --lr LR               Learning Rate. (default: 0.001)
  --batch BATCH         Batch Size. (default: 32)
  --epochs EPOCHS       Epochs. (default: 1000)
  --weights WEIGHTS     Weight path to load. If not given, training begins
                        from scratch with imagenet base weights (default: )
  --weights-no-build    Skip model building. Load saved model only. (default:
                        False)
  --summary             Display a summary of the model and exit (default:
                        False)
  --img_w IMG_W         Image Width (default: 32)
  --img_h IMG_H         Image Height (default: 32)
  --resnet-init-channel RESNET_INIT_CHANNEL
                        ResNet Initial Channel Number (default: 64)
  --distill             Perform Distillation (default: False)
  --teacher TEACHER     Teacher Model Path (default: )
  --skip-teacher-eval   Skip Teacher Evaluation on Distillation (default:
                        False)
  --temperature TEMPERATURE
                        Soft Label Temperature (default: 2.0)
  --self-distill        Training by Self-Distillation (default: False)
  --no-tensorboard      Skip running tensorboard (default: False)
  --no-tensorboard-writing
                        Skip writing tensorboard (default: False)
  --tboard-root TBOARD_ROOT
                        Tensorboard Log Root. Set this to 'no' will disable
                        writing tensorboards (default: ./export)
  --tboard-host TBOARD_HOST
                        Tensorboard Host Address (default: 0.0.0.0)
  --tboard-port TBOARD_PORT
                        TensorBoard Port Number (default: 6006)
  --tboard-profile TBOARD_PROFILE
                        Tensorboard Profiling (0: No Profile) (default: 0)
  --tboard-update-freq TBOARD_UPDATE_FREQ
                        Tensorboard Update Frequency. (epoch, batch) (default:
                        epoch)
  --logistic-hidden-ratio LOGISTIC_HIDDEN_RATIO
                        Logistic Model Hidden Neuron Ratio. (default: 0.5)
  --debug               Debugging Mode (default: False)
  --float16             Use Mixed Precision with float16 (default: False)
  --float16-dtype FLOAT16_DTYPE
                        Mixed float16 precision type. (default: mixed_float16)
  --growth-rate GROWTH_RATE
                        MicroJKNet Growth Rate (default: 12)
  --model-depth MODEL_DEPTH
                        MicroJKNet Depth (default: 3)
  --model-in-depth MODEL_IN_DEPTH
                        MicroJKNet In-Depth (default: 3)
  --compression-rate COMPRESSION_RATE
                        MicroJKNet Compression Rate (default: 2.0)
  --expansion EXPANSION
                        MicroJKNet Expansion (default: 4)
  --augment AUGMENT     Augmentation Method. (auto, album, tf, none) (default:
                        none)
  --augment-policy AUGMENT_POLICY
                        Augmentation Policy. (imagenet, cifar10, svhn)
                        (default: cifar)
  --augment-test        Apply Augmentation on Test set (default: False)
  --activation ACTIVATION
                        Activation Function (relu, swish, hswish) (default:
                        relu)
  --dropout DROPOUT     Dropout probability. (MicroJKNet Only) (default: 0.0)
  --conv CONV           Convolution Type. (conv2d, sep-conv). (MicroJKNet
                        Only) (default: conv2d)
  --data-format DATA_FORMAT
                        Data Format (channels_last, channels_first). ((batch,
                        height, width, channel), (batch, channel, height,
                        width)) (default: channels_last)
  --seed SEED           Random Seed (default: 7777)
  --prefetch            Use prefetch option for dataset (default: True)
  --test-only           Model test only (default: False)
  -en [ENSEMBLE_MODELS [ENSEMBLE_MODELS ...]], --ensemble-models [ENSEMBLE_MODELS [ENSEMBLE_MODELS ...]]
  --multi-gpu           Use multi GPU to train (default: False)
  --multi-worker MULTI_WORKER
                        Worker number of set_inter_op_parallelism_threads
                        (default: 8)
  --save-metric SAVE_METRIC
                        Auto model save metric (default:
                        val_geometric_f1score)
  --metric-type METRIC_TYPE
                        Metric type (loss, score) (default: score)
  --tta                 Use Test Time Augmentation (default: False)
  --n-tta N_TTA         Number of augmentation in TTA (default: 3)
  --tta-softmax TTA_SOFTMAX
                        Use softmax to predict class in TTA (default: True)
  --no-tta-softmax      No use softmax to predict class in TTA. Intead, use
                        sum of softmaxes (default: True)
  --tta-mp              Use Multiprocess for TTA pipeline (default: False)
Process finished with exit code 0
```
