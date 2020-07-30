import sys
import tensorflow as tf
import efficientnet.tfkeras as efn
import argparse
from tfhelper.tensorboard import get_tf_callbacks, run_tensorboard, wait_ctrl_c
from tfhelper.gpu import allow_gpu_memory_growth
from tfhelper.metrics import F1ScoreMetric
from models import resnet, DistillationModel, SelfDistillationModel, microjknet, activations, logistic, ensemble_model, TTAModel
import json
import pandas as pd
import numpy as np

if __name__ == "__main__":

    model_dict = {
        "vgg16": tf.keras.applications.VGG16,
        "vgg19": tf.keras.applications.VGG19,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "mobilenetv1": tf.keras.applications.MobileNet,
        "effnetb0": efn.EfficientNetB0,
        "effnetb1": efn.EfficientNetB1,
        "effnetb2": efn.EfficientNetB2,
        "effnetb3": efn.EfficientNetB3,
        "effnetb4": efn.EfficientNetB4,
        "effnetb5": efn.EfficientNetB5,
        "effnetb6": efn.EfficientNetB6,
        "effnetb7": efn.EfficientNetB7,
        "effnetl2": efn.EfficientNetL2,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101v2": tf.keras.applications.ResNet101V2,
        "resnet50": tf.keras.applications.ResNet50,
        "resnet50v2": tf.keras.applications.ResNet50V2,
        "resnet152": tf.keras.applications.ResNet152,
        "resnet152v2": tf.keras.applications.ResNet152V2,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "densenet121": tf.keras.applications.DenseNet121,
        "densenet169": tf.keras.applications.DenseNet169,
        "densenet201": tf.keras.applications.DenseNet201,
        "nasnetlarge": tf.keras.applications.NASNetLarge,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
        "xception": tf.keras.applications.Xception,
        "resnet18": resnet.ResNet18,
        "resnet10": resnet.ResNet10,
        "microjknet": microjknet.MicroJKNet,
        "logistic": logistic.MiniModel,
        "ensemble": ensemble_model.EnsembleModel
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-conf", default="./conf/dataset_conf.json", help="Dataset Configuration Path")
    parser.add_argument("--dataset-lib", default="../aihub_dataset", help="Dataset Library Path")
    parser.add_argument("--model", default="resnet18", type=str, help="Model Name. Supported Models: ({}).".format(
        ", ".join(list(model_dict.keys()))))
    parser.add_argument("--unfreeze", default=0, type=int, help="A number unfreeze layer. 0: Freeze all. -1: Unfreeze all.")
    parser.add_argument("--top-layer", default="gap", type=str, help="Top Layer Type(gap: 1x1conv->GAP, dense: Flatten->Fully Connected")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate.")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
    parser.add_argument("--weights", default="", type=str, help="Weight path to load. If not given, training begins from scratch with imagenet base weights")
    parser.add_argument("--weights-no-build", default=False, action='store_true', help="Skip model building. Load saved model only.")
    parser.add_argument("--summary", dest="summary", action="store_true", default=False, help="Display a summary of the model and exit")
    parser.add_argument("--img_w", default=64, type=int, help="Image Width")
    parser.add_argument("--img_h", default=48, type=int, help="Image Height")
    parser.add_argument("--resnet-init-channel", default=64, type=int, help="ResNet Initial Channel Number")
    parser.add_argument("--distill", default=False, action='store_true', help="Perform Distillation")
    parser.add_argument("--teacher", default="", type=str, help="Teacher Model Path")
    parser.add_argument("--skip-teacher-eval", default=False, action='store_true', help="Skip Teacher Evaluation on Distillation")
    parser.add_argument("--temperature", default=2.0, type=float, help="Soft Label Temperature")
    parser.add_argument("--self-distill", default=False, action='store_true', help="Training by Self-Distillation")
    parser.add_argument("--no-tensorboard", default=False, action='store_true', help="Skip running tensorboard")
    parser.add_argument("--no-tensorboard-writing", default=False, action='store_true', help="Skip writing tensorboard")
    parser.add_argument("--tboard-root", default="./export", type=str, help="Tensorboard Log Root. Set this to 'no' will disable writing tensorboards")
    parser.add_argument("--tboard-host", default="0.0.0.0", type=str, help="Tensorboard Host Address")
    parser.add_argument("--tboard-port", default=6006, type=int, help="TensorBoard Port Number")
    parser.add_argument("--tboard-profile", default=0, type=int, help="Tensorboard Profiling (0: No Profile)")
    parser.add_argument("--tboard-update-freq", default="epoch", type=str, help="Tensorboard Update Frequency. (epoch, batch)")
    parser.add_argument("--logistic-hidden-ratio", default=0.5, type=float, help="Logistic Model Hidden Neuron Ratio.")
    parser.add_argument("--debug", default=False, action='store_true', help="Debugging Mode")
    parser.add_argument("--float16", default=False, action='store_true', help="Use Mixed Precision with float16")
    parser.add_argument("--float16-dtype", default='mixed_float16', type=str, help="Mixed float16 precision type.")
    parser.add_argument("--growth-rate", default=12, type=int, help="MicroJKNet Growth Rate")
    parser.add_argument("--model-depth", default=3, type=int, help="MicroJKNet Depth")
    parser.add_argument("--model-in-depth", default=3, type=int, help="MicroJKNet In-Depth")
    parser.add_argument("--compression-rate", default=2.0, type=float, help="MicroJKNet Compression Rate")
    parser.add_argument("--expansion", default=4, type=int, help="MicroJKNet Expansion")
    parser.add_argument("--augment", default="none", type=str, help="Augmentation Method. (auto, album, tf, none)")
    parser.add_argument("--augment-policy", default="imagenet", type=str, help="Augmentation Policy. (imagenet, cifar10, svhn)")
    parser.add_argument("--augment-test", default=False, action='store_true', help="Apply Augmentation on Test set")
    parser.add_argument("--activation", default="relu", type=str, help="Activation Function (relu, swish, hswish)")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability. (MicroJKNet Only)")
    parser.add_argument("--conv", default="conv2d", type=str, help="Convolution Type. (conv2d, sep-conv). (MicroJKNet Only)")
    parser.add_argument("--load-all", default=False, action='store_true', help="Loading All Dataset into memory.")
    parser.add_argument("--reduce-dataset-ratio", default=1.0, type=float, help="Reducing dataset image numbers. (0.0 ~ 1.0)")
    parser.add_argument("--data-format", default="channels_last", type=str, help="Data Format (channels_last, channels_first). ((batch, height, width, channel), (batch, channel, height, width))")
    parser.add_argument("--seed", default=7777, type=int, help="Random Seed")
    parser.add_argument("--prefetch", default=True, dest="prefetch", action='store_true', help="Use prefetch option for dataset")
    parser.add_argument("--use-cache", default=True, dest="use_cache", action='store_true', help="Use prefetch option for dataset")
    parser.add_argument("--no-prefetch", dest="prefetch", action='store_false', help="No use prefetch option for dataset")
    parser.add_argument("--no-cache", dest="use_cache", action='store_false', help="No use cache option for dataset")
    parser.add_argument("--test-only", default=False, action='store_true', help="Model test only")
    parser.add_argument("-en", "--ensemble-models", nargs="*")
    parser.add_argument("--multi-gpu", default=False, action='store_true', help="Use multi GPU to train")
    parser.add_argument("--multi-worker", default=8, type=int, help="Worker number of set_inter_op_parallelism_threads")
    parser.add_argument("--save-metric", default="val_geometric_f1score", help="Auto model save metric")
    parser.add_argument("--metric-type", default="score", help="Metric type (loss, score)")
    parser.add_argument("--tta", default=False, action='store_true', help="Use Test Time Augmentation")
    parser.add_argument("--n-tta", default=3, type=int, help="Number of augmentation in TTA")
    parser.add_argument("--tta-softmax", dest="tta_softmax", default=True, help="Use softmax to predict class in TTA")
    parser.add_argument("--no-tta-softmax", dest="tta_softmax", action='store_false', help="No use softmax to predict class in TTA. Intead, use sum of softmaxes")
    parser.add_argument("--tta-mp", default=False, action='store_true', help="Use Multiprocess for TTA pipeline")

    args = parser.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(args.multi_worker)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    sys.path.extend([args.dataset_lib])

    from dataset.tfkeras import KProductsTFGenerator, KProductsTFGeneratorTTA
    from dataset.tfkeras import preprocessing
    from dataset import augment

    with open(args.dataset_conf, 'r', encoding='UTF8') as f:
        dataset_config = json.load(f)

    train_annotation = pd.read_csv(dataset_config['train_annotation'])
    test_annotation = pd.read_csv(dataset_config['test_annotation'])

    n_classes = len(dataset_config['label_dict'])

    if args.reduce_dataset_ratio < 1.0:
        train_annotation = train_annotation.sample(n=int(train_annotation.shape[0] * args.reduce_dataset_ratio), random_state=args.seed).reset_index(drop=True)
        n_test_by_class = np.ceil(test_annotation.shape[0] * args.reduce_dataset_ratio / n_classes).astype(np.int)

        t_annot = [test_annotation.query("{} == '{}'".format(dataset_config['class_key'], dataset_config['label_dict'][str(i)])) for i in range(len(dataset_config['label_dict']))]
        test_annotation = pd.concat([annot.sample(n=min(max(n_test_by_class, 1), annot.shape[0]), random_state=args.seed)
                                   for annot in t_annot])

    # Setting the model
    TargetModel = None

    if args.model in model_dict.keys():
        TargetModel = model_dict[args.model]

    if TargetModel is None and args.weights_no_build is False:
        print("Supported Model List")
        for i, model_name in enumerate(model_dict.keys()):
            print("{:02d}: {}".format(i+1, model_name))
        exit(0)

    devices = None if args.multi_gpu else ['/gpu:0']
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    print("Device number: {}".format(strategy.num_replicas_in_sync))

    # Setting model parameters
    kwargs = {'input_shape': (args.img_h, args.img_w, 3), 'include_top': False, 'weights': 'imagenet'}
    append_top_layer = True
    try:
        # Custom ResNet Model Parameter
        if issubclass(TargetModel, resnet.ResNet):
            kwargs['float16'] = args.float16
            kwargs['float16_dtype'] = args.float16_dtype
            kwargs['init_channel'] = args.resnet_init_channel
            kwargs['n_classes'] = n_classes
            kwargs['include_top'] = True
            append_top_layer = False
        # Custom MicroJKNet Model Parameter
        elif issubclass(TargetModel, microjknet.MicroJKNet):
            kwargs['float16'] = args.float16
            kwargs['float16_dtype'] = args.float16_dtype
            kwargs['growth_rate'] = args.growth_rate
            kwargs['depth'] = args.model_depth
            kwargs['in_depth'] = args.model_in_depth
            kwargs['expansion'] = args.expansion
            kwargs['n_classes'] = n_classes
            kwargs['compression_rate'] = args.compression_rate
            kwargs['Activation'] = activations.Hswish if args.activation == 'hswish' else activations.Swish if args.activation == 'swish' else tf.keras.layers.ReLU
            kwargs['p_drop'] = args.dropout
            kwargs['Conv'] = tf.keras.layers.SeparableConv2D if args.conv == "sep-conv" else tf.keras.layers.Conv2D
            kwargs['data_format'] = args.data_format
            args.model += f"({args.growth_rate},{args.model_depth},{args.model_in_depth},{args.expansion},{args.compression_rate},{args.activation})"
            kwargs.pop("include_top")
            kwargs.pop("weights")
            append_top_layer = False
        elif issubclass(TargetModel, logistic.MiniModel):
            # Custom Logistic Regression Model Parameter
            kwargs['float16'] = args.float16
            kwargs['float16_dtype'] = args.float16_dtype
            kwargs['n_classes'] = n_classes
            kwargs['Activation'] = activations.Hswish if args.activation == 'hswish' else activations.Swish if args.activation == 'swish' else tf.keras.layers.ReLU
            kwargs['hidden_ratio'] = args.logistic_hidden_ratio
            kwargs['data_format'] = args.data_format
            kwargs.pop("include_top")
            kwargs.pop("weights")
            append_top_layer = False
        elif issubclass(TargetModel, ensemble_model.EnsembleModel):
            ensemble_models = [tf.keras.models.load_model(path) for path in args.ensemble_models]
            kwargs['models'] = ensemble_models
            kwargs.pop("include_top")
            kwargs.pop("weights")
            append_top_layer = False
    except:
        pass

    # Build Model
    with strategy.scope():
        if args.weights_no_build:
            model = tf.keras.models.load_model(args.weights)
            args.data_format = "channels_last" if model.input.shape[-1] == 3 else "channels_first"
            args.img_h = model.input.shape[1] if args.data_format == "channels_last" else model.input.shape[2]
            args.img_w = model.input.shape[2] if args.data_format == "channels_last" else model.input.shape[3]
            args.unfreeze = len(model.layers) if args.unfreeze == 0 else args.unfreeze
            append_top_layer = False
        else:
            target_model = TargetModel(**kwargs)
            model = target_model

            if type(model) != tf.keras.models.Model:
                # Custom Model require to call build_model()
                dtype = model.dtype
                model = model.build_model()
                args.unfreeze = len(model.layers) if args.unfreeze == 0 else args.unfreeze
                args.model = args.model + "_custom"
            else:
                dtype = tf.float32

        # Freeze / Unfreeze Layers
        if args.unfreeze == 0:
            model.trainable = False
        elif args.unfreeze < 0:
            model.trainable = True
        else:
            model.trainable = True
            for i in range(0, len(model.layers)-args.unfreeze):
                model.layers[i].trainable = False

        # Append Custom Top Layer if needed.
        if append_top_layer:
            if args.top_layer == "gap":
                conv2d = tf.keras.layers.Conv2D(n_classes, 1, padding='SAME', activation=None, dtype=dtype)(model.output)
                conv2d = tf.keras.layers.BatchNormalization(dtype=dtype)(conv2d)
                conv2d = tf.keras.layers.ReLU(name="final_block_activation", dtype=dtype)(conv2d)

                output = tf.keras.layers.GlobalAveragePooling2D(dtype=dtype)(conv2d)
                output = tf.keras.layers.Softmax(name="out_dense")(output)
            else:
                output = tf.keras.layers.Flatten()(model.output)
                output = tf.keras.layers.Dense(n_classes, name="final_block_activation", dtype=dtype)(output)
                output = tf.keras.layers.Softmax(name="out_dense")(output)

            n_model = tf.keras.models.Model(model.input, output)
        else:
            n_model = model

        if args.tta:
            tta = TTAModel(n_model, n_tta=args.n_tta, use_softmax=args.tta_softmax)
            n_model = tta.build_model()

    n_model.summary()

    print("=" * 50)
    print(f"{'=' * 10}   Model: {args.model}   {'=' * 10}")

    if args.summary:
        exit(0)

    # Augmentation
    augmentation_func = None
    augment_in_dtype = "pil"
    if args.augment == "auto":
        augmentation_func = augment.SVHNPolicy() if args.augment_policy == "svhn" else augment.CIFAR10Policy() if args.augment_policy == "cifar10" else augment.ImageNetPolicy()
    elif args.augment == "album":
        augmentation_func = augment.DeepInAirPolicy()
        augment_in_dtype = "numpy"
    elif args.augment == "tf":
        augmentation_func = augment.TFAugmentPolicy(data_format=args.data_format)
        augment_in_dtype = "tensor"

    with strategy.scope():
        # Dataset Generator
        preprocess_func = preprocessing.get_preprocess_by_model_name(args.model)

        generator_args = [train_annotation, dataset_config['label_dict'], dataset_config['dataset_root']]
        kwargs = {
            "shuffle": True,
            "image_size": (args.img_h, args.img_w),
            "augment_func": augmentation_func,
            "augment_in_dtype": augment_in_dtype,
            "preprocess_func": preprocess_func,
            "prefetch": args.prefetch,
            "use_cache": args.use_cache,
            "load_all": args.load_all,
            "data_format": args.data_format
        }
        if args.tta:
            kwargs['n_tta'] = args.n_tta
            kwargs['multiprocess'] = args.tta_mp
            Generator = KProductsTFGeneratorTTA
        else:
            Generator = KProductsTFGenerator

        train_gen = Generator(*generator_args, **kwargs)

        generator_args[0] = test_annotation
        kwargs['augment_func'] = augmentation_func if args.augment_test or args.tta else None
        kwargs['shuffle'] = False

        test_gen = Generator(*generator_args, **kwargs)

        train_set = train_gen.get_tf_dataset(args.batch)
        test_set = test_gen.get_tf_dataset(args.batch)

        n_train = train_gen.annotation.shape[0]
        n_test = test_gen.annotation.shape[0]

        print("n_train: {:,}, n_test: {:,}".format(n_train, n_test))

        tboard_path = args.tboard_root
        model_out_idx = -1
        geometric_f1score = F1ScoreMetric(n_classes=n_classes, debug=args.debug, name="geometric_f1score", f1_method='geometric')
        macro_f1score = F1ScoreMetric(n_classes=n_classes, debug=args.debug, name="macro_f1score", f1_method='macro')

        if args.distill:
            teacher_f_name = args.teacher.split("/")[-1]

            print(f"{'=' * 10}   Distillation   {'=' * 10}")
            print(f"{'=' * 10}   Teacher: {teacher_f_name}   {'=' * 10}")

            try:
                teacher_model = tf.keras.models.load_model(args.teacher)
            except:
                print("Loading Teacher Model Failed at {}".format(args.teacher))
                print("Please specify teacher model file path by --teacher model_path")
                exit(0)

            distiller = DistillationModel(teacher_model, n_model, temperature=args.temperature, debug=args.debug)

            out_name = n_model.output.name.split("/")[0]

            if not args.skip_teacher_eval:
                distiller.evaluate_teacher(test_set=test_set)
                print("Teacher Model Loss: {:.5f}, Accuracy: {:.5f}".format(distiller.teacher_loss, distiller.teacher_accuracy))

            distiller.build_model()
            distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))
            n_model = distiller.distill_model

            model_out_idx = 0
            save_metric = f"val_{out_name}_accuracy_student"
            metric_type = "score"
            tboard_path += "/distill_{}_to_{}_".format(teacher_f_name, args.model)
        elif args.self_distill:
            print(f"{'=' * 10}   Self-Distillation   {'=' * 10}")
            print(f"{'=' * 10}   Target Model: {args.model}   {'=' * 10}")

            distill_param_dict = {
                tf.keras.applications.ResNet50: (['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out'], 2048, 'conv5_block3_out'),
                tf.keras.applications.ResNet50V2: (['conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu'], 2048, 'post_relu'),
                tf.keras.applications.MobileNet: (['conv_pw_3_relu', 'conv_pw_5_relu', 'conv_pw_11_relu'], 1024, 'conv_pw_13_relu'),
                tf.keras.applications.MobileNetV2: (['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu'], 1280, 'out_relu'),
                resnet.ResNet18: (['resblock_0_1_activation_1', 'resblock_1_1_activation_1', 'resblock_2_1_activation_1'], args.resnet_init_channel*8, 'resblock_3_1_activation_1'),
                resnet.ResNet10: (['resblock_0_0_activation_1', 'resblock_1_0_activation_1', 'resblock_2_0_activation_1'], args.resnet_init_channel*8, 'resblock_3_0_activation_1')
            }
            if TargetModel not in distill_param_dict.keys():
                print("Current Supported Models")
                support_models = list(distill_param_dict.keys())
                print(", ".join([support_models[i].__name__ for i in range(len(support_models))]))

            out_layer_names, final_n_filter, final_feat_layer_name = distill_param_dict[TargetModel]
            self_distiller = SelfDistillationModel(n_model, out_layer_names, final_n_filter, final_feat_layer_name, temperature=args.temperature)
            self_distiller.build_model()
            self_distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

            out_name = n_model.output.name.split("/")[0]

            model_out_idx = 0
            n_model = self_distiller.distill_model
            save_metric = f"val_{out_name}_metric_out_accuracy"
            metric_type = "score"
            tboard_path += "/self_distill_{}_".format(args.model)
        else:
            print(f"{'=' * 10}   Base Model Training   {'=' * 10}")
            print(f"{'=' * 10}   Target Model: {args.model}   {'=' * 10}")

            n_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                            loss='sparse_categorical_crossentropy', metrics=['accuracy', geometric_f1score, macro_f1score])
            save_metric = args.save_metric
            metric_type = args.metric_type
            tboard_path += "/{}_".format(args.model)

        if args.weights != "":
            if args.tta:
                tta.load_weights(args.weights)
            else:
                n_model.load_weights(args.weights)

        if args.test_only:
            if args.model.startswith("ensemble"):
                target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='sparse_categorical_crossentropy')
                target_model.evaluate(test_set, eval_ensemble=True)
                target_model.evaluate(test_set, eval_ensemble=False)
            else:
                n_model.evaluate(test_set)
            exit()

        tboard_callback = False if args.no_tensorboard_writing else True

        y_test = np.array([test_gen.reverse_label[y] for y in test_annotation[dataset_config['class_key']].values])
        callbacks, tboard_root = get_tf_callbacks(tboard_path, tboard_callback=True, tboard_profile_batch=args.tboard_profile,
                                                  tboard_update_freq=args.tboard_update_freq,
                                                  confuse_callback=tboard_callback, test_dataset=test_set, save_metric=save_metric, model_out_idx=model_out_idx,
                                                  label_info=list(dataset_config['label_dict'].values()), y_test=y_test,
                                                  modelsaver_callback=True, save_file_name=args.model, metric_type=metric_type,
                                                  save_func=tta.save if args.tta else None,
                                                  earlystop_callback=False,
                                                  sparsity_callback=tboard_callback, sparsity_threshold=0.05)

        if not args.no_tensorboard:
            run_tensorboard(tboard_root, host=args.tboard_host, port=args.tboard_port)

        n_model.fit(train_set, epochs=args.epochs, validation_data=test_set, callbacks=callbacks)

    if not args.no_tensorboard:
        wait_ctrl_c()

