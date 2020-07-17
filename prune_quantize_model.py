import tensorflow as tf
from cifar_generator import CifarGenerator
import efficientnet.tfkeras as efn
import tensorflow_model_optimization as tfmot
import numpy as np
import json
from tfhelper.tflite import keras_model_to_tflite, evaluate_tflite_interpreter
from tqdm import tqdm
import argparse
import datetime
from tfhelper.tensorboard import run_tensorboard, wait_ctrl_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Saved model path")
    parser.add_argument("--config", type=str, default="./quant_conf.json", help="Configuration file path. (Default: ./quant_conf.json)")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size. (Default: 8)")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs. (Default: 10)")
    parser.add_argument("--validate", dest="validate", action="store_true", default=False, help="Run Validation. (Default: False)")
    parser.add_argument("--is", default=0.5, type=float, help="Initial sparsity for pruning. (Default: 0.5)")
    parser.add_argument("--fs", default=0.8, type=float, help="Final sparsity for pruning. (Default: 0.8)")
    # parser.add_argument("--prune", dest="prune", action="store_true", default=False, help="Run Pruning. (Default: False)")

    args = parser.parse_args()

    file_name = args.path.split("/")[-1]
    file_root = "/".join(args.path.split("/")[:-1])
    model_name = file_name.split("_")[0]
    baseline_acc = float(file_name.split("_")[-1].replace(".h5", ""))
    dataset_type = file_name.split("_")[1]

    model = tf.keras.models.load_model(args.path)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    train_gen = CifarGenerator(x_train, y_train.flatten(), augment=True, model_type=model_name)
    test_gen = CifarGenerator(x_test, y_test.flatten(), augment=False, model_type=model_name)

    train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch*2)
    test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    if args.validate:
        baseline_loss, baseline_acc = model.evaluate(test_set)

    print("Baseline accuracy: {:.3f}%".format(baseline_acc*100))

    ### Prunning START

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    num_images = x_train.shape[0]
    end_step = np.ceil(num_images / args.batch).astype(np.int32) * args.epochs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    model_for_pruning.summary()

    path_postfix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_path = f"./export/pruning_{model_name}{path_postfix}/"
    callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=tboard_path),
        ]

    run_tensorboard(tboard_path)

    model_for_pruning.fit(train_set, batch_size=args.batch, epochs=args.epochs, validation_data=test_set, callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test_set, verbose=1)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    differene_acc = model_for_pruning_accuracy - baseline_acc
    print("Baseline accuracy: {:.3f}%".format(baseline_acc * 100))
    print("Prunned Accuracy: {:.3f} ({}{:.3f})".format(model_for_pruning_accuracy*100, "+" if differene_acc > 0 else "", differene_acc * 100))
    tf.keras.models.save_model(model_for_pruning, f"{file_root}/{model_name}_pruned_{dataset_type}_{model_for_pruning_accuracy:.4f}.h5")
    tf.keras.models.save_model(model_for_export, f"{file_root}/{model_name}_pruned_for_export_{dataset_type}_{model_for_pruning_accuracy:.4f}.h5")
    ### Prunning END

    ## Quantization not performing for now
    with open(args.config, "r") as f:
        quant_conf = json.load(f)

    quant_conf["out_path"] = f"{file_root}/{model_name}_quantized_{quant_conf['quantization_type']}_{dataset_type}.tflite"

    tflite_model = keras_model_to_tflite(model_for_export, quant_conf)

    ## Evaluate Quantized Model
    tf_interpreter = tf.lite.Interpreter(model_path=quant_conf['out_path'])
    tf_interpreter.allocate_tensors()

    t_gen = test_gen()

    accuracies = []
    prediction_list = []

    with tqdm(range(x_test.shape[0]//args.batch)) as pbar:
        for i in pbar:
            test_data = [next(t_gen) for _ in range(args.batch)]

            xx_test = np.array([test_data[j][0] for j in range(len(test_data))])
            yy_test = np.array([test_data[j][1] for j in range(len(test_data))])

            accuracy, predictions = evaluate_tflite_interpreter(tf_interpreter, xx_test, yy_test)

            accuracies = np.concatenate([accuracies, [accuracy]])
            prediction_list = np.concatenate([prediction_list, predictions])

            pbar.set_description("Mean Accuracy: {:.3f}%".format(accuracies.mean()*100))

    quantized_accuracy = accuracies.mean()
    print("Baseline accuracy: {:.3f}%".format(baseline_acc * 100))
    print("Prunned Accuracy: {:.3f} ({}{:.3f}%)".format(model_for_pruning_accuracy*100, "+" if differene_acc > 0 else "", differene_acc))

    differene_acc = quantized_accuracy - baseline_acc
    print("Quantized Accuracy: {:.3f} ({}{:.3f}%)".format(quantized_accuracy * 100, "+" if differene_acc > 0 else "", differene_acc))

    wait_ctrl_c()
