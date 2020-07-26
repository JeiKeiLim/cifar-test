import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import json
from tfhelper.tflite import keras_model_to_tflite, evaluate_tflite_interpreter
from tqdm import tqdm
import argparse
import datetime
from tfhelper.tensorboard import run_tensorboard, wait_ctrl_c, get_tf_callbacks
import sys
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Saved model path")
    parser.add_argument("--model-name", default="resnet18_custom", type=str, help="Model Name for Image Pre-Process Decision Purpose")
    parser.add_argument("--dataset-lib", default="../aihub_dataset", help="Dataset Library Path")
    parser.add_argument("--dataset-conf", default="./conf/dataset_conf.json", help="Dataset Configuration Path")
    parser.add_argument("--config", type=str, default="./conf/quant_conf.json", help="Configuration file path. (Default: ./quant_conf.json)")
    parser.add_argument("--out", type=str, default="None", help="Destination of Pruned Model Path. If 'None' is given, It will store at saved model path")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size. (Default: 8)")
    parser.add_argument("--q-aware-train", default=False, action='store_true', help="Run Quantization Aware Training")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs. (Default: 10)")
    parser.add_argument("--validate", dest="validate", action="store_true", default=False, help="Run Validation. (Default: False)")
    parser.add_argument("--baseline-acc", default=float('nan'), type=float, help="Base Line Accuracy. If both baseline-acc and baseline-loss are not given, model evaluation is performed")
    parser.add_argument("--baseline-loss", default=float('nan'), type=float, help="Base Line Loss. If both baseline-acc and baseline-loss are not given, model evaluation is performed")
    parser.add_argument("--tboard-root", default="./export", type=str, help="Tensorboard Log Root. Set this to 'no' will disable writing tensorboards")
    parser.add_argument("--tboard-host", default="0.0.0.0", type=str, help="Tensorboard Host Address")
    parser.add_argument("--tboard-port", default=6006, type=int, help="TensorBoard Port Number")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate.")
    parser.add_argument("--seed", default=7777, type=int, help="Random Seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    sys.path.extend([args.dataset_lib])

    from dataset.tfkeras import KProductsTFGenerator
    from dataset.tfkeras import preprocessing

    file_name = args.path.split("/")[-1]
    file_root = "/".join(args.path.split("/")[:-1])
    model_name = file_name.split("_")[0]

    model = tf.keras.models.load_model(args.path)

    with open(args.dataset_conf, 'r') as f:
        dataset_config = json.load(f)

    with open(args.config, "r") as f:
        quant_conf = json.load(f)

    train_annotation = pd.read_csv(dataset_config['train_annotation'])
    test_annotation = pd.read_csv(dataset_config['test_annotation'])

    img_h, img_w = model.input.shape[1:3]

    train_gen = KProductsTFGenerator(train_annotation, dataset_config['label_dict'], dataset_config['dataset_root'],
                                     shuffle=True, image_size=(img_h, img_w),
                                     augment_func=None,
                                     preprocess_func=preprocessing.get_preprocess_by_model_name(args.model_name),
                                     seed=args.seed)
    test_gen = KProductsTFGenerator(test_annotation, dataset_config['label_dict'], dataset_config['dataset_root'],
                                     shuffle=False, image_size=(img_h, img_w),
                                     preprocess_func=preprocessing.get_preprocess_by_model_name(args.model_name),
                                    seed=args.seed)

    train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch*2)
    test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    n_train = train_gen.annotation.shape[0]
    n_test = test_gen.annotation.shape[0]

    model.summary()

    print("n_train: {:,}, n_test: {:,}".format(n_train, n_test))

    if np.isnan(args.baseline_acc) and np.isnan(args.baseline_loss):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        baseline_loss, baseline_acc = model.evaluate(test_set, steps=(n_test//args.batch), verbose=1)
    else:
        baseline_acc = args.baseline_acc if args.baseline_acc > 0 else float('nan')
        baseline_loss = args.baseline_loss if args.baseline_loss > 0 else float('nan')

    print("Baseline Loss, Accuracy: {:.8f}, {:.3f}%, ".format(baseline_loss, baseline_acc*100))

    if args.q_aware_train:
        quantize_model = tfmot.quantization.keras.quantize_model

        q_aware_model = quantize_model(model)
        q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])
        q_aware_model.summary()

        tboard_callback = False if args.no_tensorboard_writing else True

        callbacks, tboard_root = get_tf_callbacks(f"{args.tboard_root}/quantization_", tboard_callback=tboard_callback,
                                                  confuse_callback=True, test_dataset=test_set,
                                                  label_info=list(dataset_config['label_dict'].values()),
                                                  modelsaver_callback=False,
                                                  earlystop_callback=False,
                                                  sparsity_callback=False, sparsity_threshold=0.05)

        run_tensorboard(tboard_root, host=args.tboard_host, port=args.tboard_port)

        q_aware_model.fit(train_set, epochs=args.epochs, validation_data=test_set)
        model = q_aware_model

    if args.out == "None":
        args.out = args.path

    out_path_split = args.out.split('/')
    out_root, out_file = "/".join(out_path_split[:-1]), out_path_split[-1]
    ext_idx = out_file.rfind('.')

    out_file = out_file[:ext_idx] if ext_idx > 0 else out_file

    quant_conf["out_path"] = f"{out_root}/{out_file}_quant_{quant_conf['quantization_type']}.tflite"

    keras_model_to_tflite(model, quant_conf)

    ## Evaluate Quantized Model
    tf_interpreter = tf.lite.Interpreter(model_path=quant_conf['out_path'])
    tf_interpreter.allocate_tensors()

    t_gen = test_gen()

    accuracies = []
    prediction_list = []

    test_iterator = test_set.as_numpy_iterator()

    with tqdm(range(n_test//args.batch)) as pbar:
        for i in pbar:
            img, label = next(test_iterator)
            accuracy, predictions = evaluate_tflite_interpreter(tf_interpreter, img, label)

            accuracies = np.concatenate([accuracies, [accuracy]])
            prediction_list = np.concatenate([prediction_list, predictions])

            pbar.set_description("Mean Accuracy: {:.3f}%".format(accuracies.mean()*100))

    quantized_accuracy = accuracies.mean()
    print("Baseline accuracy: {:.3f}%".format(baseline_acc * 100))

    differene_acc = quantized_accuracy - baseline_acc
    print("Quantized Accuracy: {:.3f} ({}{:.3f}%)".format(quantized_accuracy * 100, "+" if differene_acc > 0 else "", differene_acc))

    if args.q_aware_train:
        wait_ctrl_c()