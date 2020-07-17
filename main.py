import tensorflow as tf
from cifar_generator import Cifar100Generator
import tensorflow_model_optimization as tfmot
import argparse
import numpy as np
import time


def compute_sparsity(model_, sparse_threshold=0.05):
    sparsities = np.zeros(len(model_.layers))

    for i in range(sparsities.shape[0]):
        if len(model_.layers[i].weights) < 1:
            sparsities[i] = np.nan
            continue

        sparse_index = np.argwhere(np.logical_and(model_.layers[i].weights[0].numpy().flatten() < sparse_threshold,
                                                model_.layers[i].weights[0].numpy().flatten() > -sparse_threshold))

        sparsities[i] = sparse_index.shape[0] / np.prod(model_.layers[i].weights[0].shape)

    mean_sparsity = np.nanmean(sparsities)
    return sparsities, mean_sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Saved model path")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate. (Default: 0.001)")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size. (Default: 8)")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs. (Default: 10)")
    parser.add_argument("--sparse", default=0.1, type=float, help="Sparse Threshold. (Default: 0.1)")
    parser.add_argument("--cutting", default=0.999, type=float, help="Cutting Sparse Threshold. (Default: 0.999)")
    parser.add_argument("--validate", dest="validate", action="store_true", default=False, help="Run Validation. (Default: False)")

    args = parser.parse_args()

    file_name = args.path.split("/")[-1]
    file_root = "/".join(args.path.split("/")[:-1])
    model_name = file_name.split("_")[0]
    baseline_acc = float(file_name.split("_")[-1].replace(".h5", ""))
    dataset_type = file_name.split("_")[1]

    model = tf.keras.models.load_model(args.path)
    baseline_param_count = model.count_params()
    baseline_layer_count = len(model.layers)

    sparsities, mean_sparsity = compute_sparsity(model, sparse_threshold=args.sparse)
    idx = np.argwhere(sparsities > args.cutting).flatten()

    remove_layers = []
    in_out_connect_layers = []

    for i in range(0, len(model.layers)):
        if i in idx:
            if type(model.layers[i-1]) == tf.keras.layers.ZeroPadding2D:
                remove_layers += [model.layers[i-1], model.layers[i]]
                model._layers[i+1]._input = model._layers[i-2].output
                # in_out_connect_layers.append([model.layers[i-2], model.layers[i+1]])
            # elif type(model.layers[i+1]) == tf.keras.layers.Add:
            #     skip_flag = True
            #
            #     out_shape = model.layers[i+1].output.shape[-1]
            #     layers.pop(-1)
            #
            #     for j in (i, -1, -1):
            #         if model.layers[j].output.shape[-1] == out_shape:
            #             break
            #         layers.pop(-1)

            # elif type(model.)

    # for r_layer in remove_layers:
    #     model._layers.remove(r_layer)
    x = model.input
    for i in range(len(model.layers[1:])):
        if model.layers[i] not in remove_layers:
            x = model.layers[i](x)


    cut_model_path = f"{file_root}/{model_name}_cut_{dataset_type}.h5"
    tf.keras.models.save_model(model, cut_model_path)
    model = tf.keras.models.load_model(cut_model_path)

    cut_param_count = model.count_params()
    cut_layer_count = len(model.layers)

    print(f"Baseline Params: {baseline_param_count:,}, Baseline layers: {baseline_layer_count}")
    print(f"Cut Model Params: {cut_param_count:,}, Cut Model layers: {cut_layer_count}")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    model.summary()

    train_gen = Cifar100Generator(x_train, y_train.flatten(), augment=True, model_type=model_name)
    test_gen = Cifar100Generator(x_test, y_test.flatten(), augment=False, model_type=model_name)

    train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch*2)
    test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    s_time = time.time()
    cut_loss, cut_accuracy = model.evaluate(test_set, verbose=1)
    cut_infer_time = time.time() - s_time

    tf.keras.models.save_model(model, f"{file_root}/{model_name}_cut_{dataset_type}_{cut_accuracy:.4f}.h5", include_optimizer=False)

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(args.path)
    s_time = time.time()
    baseline_loss, baseline_acc = model.evaluate(test_set, verbose=1)
    baseline_infer_time = time.time() - s_time

    print(f"Baseline Params: {baseline_param_count:,}, Baseline layers: {baseline_layer_count}")
    print(f"Cut Model Params: {cut_param_count:,}, Cut Model layers: {cut_layer_count}")

    print("Baseline Accuracy: {:.3f}%".format(baseline_acc*100))
    print("Cut Model Accuracy: {:.3f}%".format(cut_accuracy*100))

    print("Baseline Inference Time: {:.1f}s".format(baseline_infer_time))
    print("Cut Model Inference Time: {:.1f}s".format(cut_infer_time))

    tboard_path = "./export/cut_{}".format(model_name)


