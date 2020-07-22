import tensorflow as tf
from cifar_generator import CifarGenerator
import efficientnet.tfkeras as efn
import argparse
from tfhelper.tensorboard import get_tf_callbacks, run_tensorboard, wait_ctrl_c
from tfhelper.gpu import allow_gpu_memory_growth
from models import resnet, DistillationModel, SelfDistillationModel


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
        "resnet10": resnet.ResNet10
    }
    dataset_dict = {
        "cifar10": tf.keras.datasets.cifar10,
        "cifar100": tf.keras.datasets.cifar100,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="resnet18", type=str, help="Model Name. Supported Models: ({}).".format(
        ", ".join(list(model_dict.keys()))))
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset Name. Supported Dataset: ({}).".format(", ".join(list(dataset_dict.keys()))))
    parser.add_argument("--unfreeze", default=0, type=int, help="A number unfreeze layer. 0: Freeze all. -1: Unfreeze all.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate.")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
    parser.add_argument("--weights", default="", type=str, help="Weight path to load. If not given, training begins from scratch with imagenet base weights")
    parser.add_argument("--summary", dest="summary", action="store_true", default=False, help="Display a summary of the model and exit")
    parser.add_argument("--img_w", default=224, type=int, help="Image Width")
    parser.add_argument("--img_h", default=224, type=int, help="Image Height")
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
    parser.add_argument("--debug", default=False, action='store_true', help="Debugging Mode")

    args = parser.parse_args()

    if args.dataset in dataset_dict.keys():
        (x_train, y_train), (x_test, y_test) = dataset_dict[args.dataset].load_data()
    else:
        print("Supported Dataset List")
        for i, dataset_name in enumerate(dataset_dict.keys()):
            print("{:02d}: {}".format(i+1, dataset_name))
        exit(0)

    TargetModel = None

    if args.model in model_dict.keys():
        TargetModel = model_dict[args.model]

    if TargetModel is None:
        print("Supported Model List")
        for i, model_name in enumerate(model_dict.keys()):
            print("{:02d}: {}".format(i+1, model_name))
        exit(0)

    allow_gpu_memory_growth()

    model = TargetModel(input_shape=(args.img_w, args.img_h, 3), include_top=False, weights='imagenet')
    if type(model) != tf.keras.models.Model:
        model = model.build_model()
        args.unfreeze = len(model.layers) if args.unfreeze == 0 else args.unfreeze
        args.model = args.model + "_custom"

    if args.unfreeze == 0:
        model.trainable = False
    elif args.unfreeze < 0:
        model.trainable = True
    else:
        model.trainable = True
        for i in range(0, len(model.layers)-args.unfreeze):
            model.layers[i].trainable = False

    conv2d = tf.keras.layers.Conv2D(y_train.max()+1, 1, padding='SAME', activation=None)(model.output)
    conv2d = tf.keras.layers.BatchNormalization()(conv2d)
    conv2d = tf.keras.layers.ReLU(name="final_block_activation")(conv2d)

    output = tf.keras.layers.GlobalAveragePooling2D()(conv2d)
    output = tf.keras.layers.Softmax(name="out_dense")(output)

    n_model = tf.keras.models.Model(model.input, output)

    if args.weights != "":
        n_model.load_weights(args.weights)

    n_model.summary()

    if args.summary:
        exit(0)

    train_gen = CifarGenerator(x_train, y_train.flatten(), augment=True, model_type=args.model, image_size=(args.img_w, args.img_h))
    test_gen = CifarGenerator(x_test, y_test.flatten(), augment=False, model_type=args.model, image_size=(args.img_w, args.img_h))

    train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch * 2)
    test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    print("="*50)

    tboard_path = args.tboard_root

    if args.distill:
        teacher_f_name = args.teacher.split("/")[-1]

        print(f"{'=' * 10}   Distillation   {'=' * 10}")
        print(f"{'=' * 10}   Teacher: {teacher_f_name}   {'=' * 10}")
        print(f"{'=' * 10}   Student: {args.model}   {'=' * 10}")

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

        save_metric = f"val_{out_name}_accuracy_student"
        tboard_path += "/distill_{}_to_{}_".format(args.tboard_root, teacher_f_name, args.model)
    elif args.self_distill:
        print(f"{'=' * 10}   Self-Distillation   {'=' * 10}")
        print(f"{'=' * 10}   Target Model: {args.model}   {'=' * 10}")

        distill_param_dict = {
            tf.keras.applications.ResNet50: (['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out'], 2048, 'conv5_block3_out'),
            tf.keras.applications.ResNet50V2: (['conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu'], 2048, 'post_relu'),
            tf.keras.applications.MobileNet: (['conv_pw_3_relu', 'conv_pw_5_relu', 'conv_pw_11_relu'], 1024, 'conv_pw_13_relu'),
            tf.keras.applications.MobileNetV2: (['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu'], 1280, 'out_relu'),
            resnet.ResNet18: (['resblock_0_1_activation_1', 'resblock_1_1_activation_1', 'resblock_2_1_activation_1'], 512, 'resblock_3_1_activation_1'),
            resnet.ResNet10: (['resblock_0_0_activation_1', 'resblock_1_0_activation_1', 'resblock_2_0_activation_1'], 512, 'resblock_3_0_activation_1')
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

        n_model = self_distiller.distill_model
        save_metric = f"val_{out_name}_metric_out_accuracy"
        tboard_path += "/self_distill_{}_".format(args.model)
    else:
        print(f"{'=' * 10}   Base Model Training   {'=' * 10}")
        print(f"{'=' * 10}   Target Model: {args.model}   {'=' * 10}")

        n_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        save_metric = 'val_accuracy'
        tboard_path += "/{}_".format(args.model)

    tboard_callback = False if args.no_tensorboard_writing else True

    callbacks, tboard_root = get_tf_callbacks(tboard_path, tboard_callback=tboard_callback, tboard_profile_batch=args.tboard_profile,
                                              confuse_callback=False, test_dataset=test_set, save_metric=save_metric,
                                              modelsaver_callback=True,
                                              earlystop_callback=False,
                                              sparsity_callback=True, sparsity_threshold=0.05)

    if not args.no_tensorboard:
        run_tensorboard(tboard_root, host=args.tboard_host, port=args.tboard_port)

    n_model.fit(train_set, epochs=args.epochs, validation_data=test_set, callbacks=callbacks)

    if not args.no_tensorboard:
        wait_ctrl_c()

