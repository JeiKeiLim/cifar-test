import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import json
import argparse
import datetime
from tfhelper.tensorboard import run_tensorboard, wait_ctrl_c, SparsityCallback
import sys
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help="Saved model path")
    parser.add_argument("--model-name", default="resnet18_custom", type=str, help="Model Name for Image Pre-Process Decision Purpose")
    parser.add_argument("--dataset-lib", default="../aihub_dataset", help="Dataset Library Path")
    parser.add_argument("--dataset-conf", default="./conf/dataset_conf.json", help="Dataset Configuration Path")
    parser.add_argument("--out", type=str, default="None", help="Destination of Pruned Model Path. If 'None' is given, It will store at saved model path")
    parser.add_argument("--batch", default=8, type=int, help="Batch Size.")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs.")
    parser.add_argument("--baseline-acc", default=float('nan'), type=float, help="Base Line Accuracy. If both baseline-acc and baseline-loss are not given, model evaluation is performed")
    parser.add_argument("--baseline-loss", default=float('nan'), type=float, help="Base Line Loss. If both baseline-acc and baseline-loss are not given, model evaluation is performed")
    parser.add_argument("--init-sparsity", default=0.5, type=float, help="Initial sparsity for pruning. If -1 is given, Initial sparsity is computed by sparsity-threshold.")
    parser.add_argument("--final-sparsity", default=0.8, type=float, help="Final sparsity for pruning.")
    parser.add_argument("--sparsity-threshold", default=0.05, type=float, help="Sparsity threshold value to find sparsity levels on each layer.")
    parser.add_argument("--reduce-dataset-ratio", default=1.0, type=float, help="Reducing dataset image numbers. (0.0 ~ 1.0)")
    parser.add_argument("--tboard-root", default="./export", type=str, help="Tensorboard Log Root. Set this to 'no' will disable writing tensorboards")
    parser.add_argument("--tboard-host", default="0.0.0.0", type=str, help="Tensorboard Host Address")
    parser.add_argument("--tboard-port", default=6006, type=int, help="TensorBoard Port Number")
    parser.add_argument("--seed", default=7777, type=int, help="Random Seed")

    args = parser.parse_args()
    np.seed(args.seed)
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

    train_annotation = pd.read_csv(dataset_config['train_annotation'])
    test_annotation = pd.read_csv(dataset_config['test_annotation'])

    if args.reduce_dataset_ratio < 1.0:
        train_annotation = train_annotation.sample(n=int(train_annotation.shape[0] * args.reduce_dataset_ratio), random_state=args.seed).reset_index(drop=True)
        test_annotation = test_annotation.sample(n=int(test_annotation.shape[0] * args.reduce_dataset_ratio), random_state=args.seed).reset_index(drop=True)

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

    train_set = train_gen.get_tf_dataset(args.batch, shuffle=True, reshuffle=True, shuffle_size=args.batch * 2)
    test_set = test_gen.get_tf_dataset(args.batch, shuffle=False)

    n_train = train_gen.annotation.shape[0]
    n_test = test_gen.annotation.shape[0]

    print("n_train: {:,}, n_test: {:,}".format(n_train, n_test))

    if np.isnan(args.baseline_acc) and np.isnan(args.baseline_loss):
        baseline_loss, baseline_acc = model.evaluate(test_set, steps=(n_test//args.batch), verbose=1)
    else:
        baseline_acc = args.baseline_acc if args.baseline_acc > 0 else float('nan')
        baseline_loss = args.baseline_loss if args.baseline_loss > 0 else float('nan')

    print("Baseline Loss, Accuracy: {:.8f}, {:.3f}%, ".format(baseline_loss, baseline_acc*100))
    ### Prunning START

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    end_step = np.ceil(n_train / args.batch).astype(np.int32) * args.epochs

    if args.init_sparsity < 0:
        sparsity_calculater = SparsityCallback(None, sparsity_threshold=args.sparsity_threshold)
        sparsity_calculater.model = model
        sparsities = sparsity_calculater.compute_sparsity()
        sparsities = sparsities[np.logical_and(~np.isnan(sparsities), sparsities != 0.0)]
        init_sparsity = sparsities.mean()
    else:
        init_sparsity = args.init_sparsity

    print("Initial Sparsity: {:.5f}".format(init_sparsity))
    print("Target Sparsity: {:.5f}".format(args.final_sparsity))
    print("Total Steps: {}".format(end_step))

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=init_sparsity,
                                                                 final_sparsity=args.final_sparsity,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

    model_for_pruning.summary()

    path_postfix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_path = f"{args.tboard_root}/pruning_{model_name}{path_postfix}/"
    callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=tboard_path),
        ]

    run_tensorboard(tboard_path, host=args.tboard_host, port=args.tboard_port)

    model_for_pruning.fit(train_set, epochs=args.epochs, validation_data=test_set, callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test_set, steps=(n_test//args.batch), verbose=1)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    differene_acc = model_for_pruning_accuracy - baseline_acc
    print("Baseline accuracy: {:.3f}%".format(baseline_acc * 100))
    print("Prunned Accuracy: {:.3f} ({}{:.3f})".format(model_for_pruning_accuracy*100, "+" if differene_acc > 0 else "", differene_acc * 100))

    if args.out == "None":
        args.out = args.path

    out_path_split = args.out.split('/')
    out_root, out_file = "/".join(out_path_split[:-1]), out_path_split[-1]
    ext_idx = out_file.rfind('.')

    out_file = out_file[:ext_idx] if ext_idx > 0 else out_file

    tf.keras.models.save_model(model_for_pruning, f"{out_root}/{out_file}_{model_for_pruning_accuracy:.5f}.h5")
    tf.keras.models.save_model(model_for_export, f"{out_root}/{out_file}_pruned_export_{model_for_pruning_accuracy:.5f}.h5")

    ### Prunning END

    wait_ctrl_c()
