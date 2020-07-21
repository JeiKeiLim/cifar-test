import tensorflow as tf
from cifar_generator import CifarGenerator
import efficientnet.tfkeras as efn
import argparse
from tfhelper.tensorboard import get_tf_callbacks, run_tensorboard, wait_ctrl_c
from tfhelper.gpu import allow_gpu_memory_growth
from models import resnet, DistillationModel, DistillationGenerator


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", type=str, help="Model Name. Supported Models: ({}). (Default: resnet18)".format(
        ", ".join(list(model_dict.keys()))))
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset Name. Supported Dataset: ({}). (Default: cifar10)".format(", ".join(list(dataset_dict.keys()))))
    parser.add_argument("--unfreeze", default=0, type=int, help="A number unfreeze layer. 0: Freeze all. -1: Unfreeze all. (Default: 0)")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate. (Default: 0.001)")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size. (Default: 8)")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs. (Default: 1000)")
    parser.add_argument("--weights", default="", type=str, help="Weight path to load. If not given, training begins from scratch with imagenet base weights")
    parser.add_argument("--summary", dest="summary", action="store_true", default=False, help="Display a summary of the model and exit")
    parser.add_argument("--img_w", default=224, type=int, help="Image Width (Default: 224)")
    parser.add_argument("--img_h", default=224, type=int, help="Image Height (Default: 224)")
    parser.add_argument("--distill", default=False, action='store_true', help="Perform Distillation")
    parser.add_argument("--teacher", default="", type=str, help="Teacher Model Path")
    parser.add_argument("--temperature", default=2.0, type=float, help="Soft Label Temperature")

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
    conv2d = tf.keras.layers.ReLU()(conv2d)

    output = tf.keras.layers.GlobalAveragePooling2D()(conv2d)
    output = tf.keras.layers.Softmax()(output)

    n_model = tf.keras.models.Model(model.input, output)

    if args.weights != "":
        n_model.load_weights(args.weights)

    n_model.summary()

    if args.summary:
        exit(0)

    train_gen = CifarGenerator(x_train, y_train.flatten(), augment=True, model_type=args.model, image_size=(args.img_w, args.img_h))
    test_gen = CifarGenerator(x_test, y_test.flatten(), augment=False, model_type=args.model, image_size=(args.img_w, args.img_h))

    if args.distill and args.teacher != "":
        teacher_model = tf.keras.models.load_model(args.teacher)
        distillation_model = DistillationModel(teacher_model, n_model, temperature=args.temperature)

        teacher_test_gen = DistillationGenerator(test_gen(), test_gen.data.shape[0], teacher_model, from_teacher=False)
        test_set = teacher_test_gen.get_tf_dataset(args.batch, shuffle=False)

        teacher_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        distillation_model.evaluate_teacher(test_set=test_set)

        print("Teacher Model Loss: {:.5f}, Accuracy: {:.5f}".format(distillation_model.teacher_loss, distillation_model.teacher_accuracy))

        distillation_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))
        n_model = distillation_model.student_model

        train_gen = DistillationGenerator(train_gen(), train_gen.data.shape[0], teacher_model, from_teacher=True)
        test_gen = DistillationGenerator(test_gen(), test_gen.data.shape[0], teacher_model, from_teacher=True)

        train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch*2)
        test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)
        save_metric = 'val_metric_accuracy'
    else:
        n_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        save_metric = 'val_accuracy'

        train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch*2)
        test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    tboard_path = "./export/{}".format(args.model)

    callbacks, tboard_root = get_tf_callbacks(tboard_path, tboard_callback=True,
                                              confuse_callback=False, test_dataset=test_set, save_metric=save_metric,
                                              modelsaver_callback=True,
                                              earlystop_callback=False,
                                              sparsity_callback=True, sparsity_threshold=0.05)

    run_tensorboard(tboard_root)

    n_model.fit(train_set, epochs=args.epochs, validation_data=test_set, callbacks=callbacks)

    wait_ctrl_c()
