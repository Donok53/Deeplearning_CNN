# hw3_cnn_fer2013.py
# FER2013 CNN classification for HW#3
# - Loads fer2013.csv
# - Trains several CNN settings
# - Selects best model by validation accuracy
# - Evaluates on test set
# - Saves architecture summary / results / confusion matrix
# - Visualizes step-by-step feature maps

from __future__ import annotations

import os
import json
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 0. Global settings
# =========================================================
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48
NUM_CLASSES = 7
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 1. Utilities
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_pixels(pixel_string: str) -> np.ndarray:
    arr = np.fromstring(pixel_string, sep=" ", dtype=np.float32)
    arr = arr.reshape(IMG_SIZE, IMG_SIZE, 1)
    arr /= 255.0
    return arr


def load_fer2013(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    images = np.stack([parse_pixels(p) for p in df["pixels"].tolist()])
    labels = df["emotion"].astype(np.int32).to_numpy()
    usage = df["Usage"].astype(str).to_numpy()

    # Kaggle FER2013 split convention
    train_mask = usage == "Training"
    val_mask = np.isin(usage, ["PublicTest", "Public Test"])
    test_mask = np.isin(usage, ["PrivateTest", "Private Test"])

    X_train = images[train_mask]
    y_train = labels[train_mask]

    X_val = images[val_mask]
    y_val = labels[val_mask]

    X_test = images[test_mask]
    y_test = labels[test_mask]

    print("Loaded FER2013 dataset")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist["loss"], label="train loss")
    plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, hist["accuracy"], label="train acc")
    plt.plot(epochs, hist["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_save_path = save_path.replace("_loss.png", "_acc.png")
    plt.savefig(acc_save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# 2. Model
# =========================================================
def apply_activation(x: tf.Tensor, activation_name: str, layer_name: str) -> tf.Tensor:
    if activation_name == "relu":
        return tf.keras.layers.ReLU(name=layer_name)(x)
    if activation_name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.1, name=layer_name)(x)
    if activation_name == "sigmoid":
        return tf.keras.layers.Activation("sigmoid", name=layer_name)(x)
    raise ValueError(f"Unsupported activation: {activation_name}")


def conv_block(
    x: tf.Tensor,
    filters: int,
    activation: str,
    block_id: int,
    dropout_rate: float
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False, name=f"block{block_id}_conv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"block{block_id}_bn1")(x)
    x = apply_activation(x, activation, f"block{block_id}_act1")

    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False, name=f"block{block_id}_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"block{block_id}_bn2")(x)
    x = apply_activation(x, activation, f"block{block_id}_act2")

    x = tf.keras.layers.MaxPooling2D((2, 2), name=f"block{block_id}_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"block{block_id}_drop")(x)
    return x


def build_model(
    activation: str = "relu",
    fc_dropout: float = 0.5,
    use_augmentation: bool = True
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_image")
    x = inputs

    if use_augmentation:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomTranslation(0.08, 0.08),
                tf.keras.layers.RandomZoom(0.08),
            ],
            name="augmentation"
        )
        x = aug(x)

    x = conv_block(x, 32, activation, block_id=1, dropout_rate=0.25)
    x = conv_block(x, 64, activation, block_id=2, dropout_rate=0.25)
    x = conv_block(x, 128, activation, block_id=3, dropout_rate=0.30)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, use_bias=False, name="fc1")(x)
    x = tf.keras.layers.BatchNormalization(name="fc1_bn")(x)
    x = apply_activation(x, activation, "fc1_act")
    x = tf.keras.layers.Dropout(fc_dropout, name="fc1_drop")(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"FER2013_CNN_{activation}")
    return model


# =========================================================
# 3. Training / evaluation
# =========================================================
def train_one_experiment(
    config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str,
    class_weight: Dict[int, float] | None = None
) -> Dict:
    exp_name = config["name"]
    exp_dir = os.path.join(output_dir, exp_name)
    ensure_dir(exp_dir)

    model = build_model(
        activation=config["activation"],
        fc_dropout=config["fc_dropout"],
        use_augmentation=config["use_augmentation"]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    summary_path = os.path.join(exp_dir, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    ckpt_path = os.path.join(exp_dir, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n==============================")
    print(f"Start experiment: {exp_name}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"==============================")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weight if config["use_class_weight"] else None,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history, os.path.join(exp_dir, "history_loss.png"))

    best_val_acc = float(np.max(history.history["val_accuracy"]))
    best_val_loss = float(np.min(history.history["val_loss"]))

    result = {
        "name": exp_name,
        "activation": config["activation"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs_run": len(history.history["loss"]),
        "fc_dropout": config["fc_dropout"],
        "use_augmentation": config["use_augmentation"],
        "use_class_weight": config["use_class_weight"],
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "checkpoint_path": ckpt_path,
        "summary_path": summary_path
    }

    save_json(result, os.path.join(exp_dir, "result.json"))
    return result


def evaluate_best_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str
) -> Dict:
    model = tf.keras.models.load_model(model_path)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    report = classification_report(
        y_test, preds,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(y_test, preds)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(output_dir, "confusion_matrix.png"))

    result = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    save_json(result, os.path.join(output_dir, "test_result.json"))
    return result


# =========================================================
# 4. Feature map visualization
# =========================================================
def visualize_feature_maps(
    model_path: str,
    image: np.ndarray,
    true_label: int,
    save_path: str,
    layer_names: List[str] | None = None,
    max_maps: int = 8
) -> None:
    if layer_names is None:
        layer_names = ["block1_conv1", "block2_conv1", "block3_conv1"]

    model = tf.keras.models.load_model(model_path)

    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)

    sample = np.expand_dims(image, axis=0)
    feature_maps = activation_model.predict(sample, verbose=0)
    pred = model.predict(sample, verbose=0)
    pred_label = int(np.argmax(pred, axis=1)[0])

    total_rows = len(layer_names) + 1
    plt.figure(figsize=(2 * max_maps, 2 * total_rows))

    # original image
    plt.subplot(total_rows, max_maps, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Input\nT:{CLASS_NAMES[true_label]}\nP:{CLASS_NAMES[pred_label]}")
    plt.axis("off")

    for empty_col in range(2, max_maps + 1):
        plt.subplot(total_rows, max_maps, empty_col)
        plt.axis("off")

    for row_idx, fmap in enumerate(feature_maps, start=2):
        fmap = fmap[0]  # (H, W, C)
        channels = min(max_maps, fmap.shape[-1])

        for ch in range(channels):
            plt.subplot(total_rows, max_maps, (row_idx - 1) * max_maps + ch + 1)
            plt.imshow(fmap[:, :, ch], cmap="viridis")
            if ch == 0:
                plt.title(layer_names[row_idx - 2])
            plt.axis("off")

        for ch in range(channels + 1, max_maps + 1):
            plt.subplot(total_rows, max_maps, (row_idx - 1) * max_maps + ch)
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def find_default_csv() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 가장 먼저 확인할 후보들
    candidates = [
        os.path.join(base_dir, "dataset", "fer2013.csv"),
        os.path.join(base_dir, "dataset", "fer2013", "fer2013.csv"),
        os.path.join(
            base_dir,
            "dataset",
            "challenges-in-representation-learning-facial-expression-recognition-challenge",
            "fer2013",
            "fer2013",
            "fer2013.csv"
        ),
    ]

    for path in candidates:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            return norm_path

    # dataset 폴더 아래를 재귀적으로 탐색
    dataset_dir = os.path.join(base_dir, "dataset")
    if os.path.exists(dataset_dir):
        for root, _, files in os.walk(dataset_dir):
            if "fer2013.csv" in files:
                return os.path.normpath(os.path.join(root, "fer2013.csv"))

    raise FileNotFoundError(
        "Could not find fer2013.csv under the project dataset folder.\n"
        f"Searched under: {os.path.normpath(os.path.join(base_dir, 'dataset'))}"
    )


# =========================================================
# 5. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to fer2013.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hw3_outputs",
        help="Folder to save results"
    )

    args = parser.parse_args()

    if args.csv is None:
        args.csv = find_default_csv()
    else:
        args.csv = os.path.normpath(args.csv)

    args.output_dir = os.path.normpath(args.output_dir)

    print(f"Using CSV: {args.csv}")
    print(f"Output dir: {args.output_dir}")

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    ensure_dir(args.output_dir)

    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(args.csv)
    class_weight = compute_class_weights(y_train)
    print("Class weights:", class_weight)

    experiments = [
        {
            "name": "exp1_relu_lr1e3_aug",
            "activation": "relu",
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 50,
            "fc_dropout": 0.50,
            "use_augmentation": True,
            "use_class_weight": False
        },
        {
            "name": "exp2_relu_lr1e3_aug_classw",
            "activation": "relu",
            "lr": 1e-3,
            "batch_size": 64,
            "epochs": 50,
            "fc_dropout": 0.50,
            "use_augmentation": True,
            "use_class_weight": True
        },
        {
            "name": "exp3_relu_lr3e4_aug_classw",
            "activation": "relu",
            "lr": 3e-4,
            "batch_size": 64,
            "epochs": 60,
            "fc_dropout": 0.50,
            "use_augmentation": True,
            "use_class_weight": True
        },
        {
            "name": "exp4_lrelu_lr3e4_aug_classw",
            "activation": "leaky_relu",
            "lr": 3e-4,
            "batch_size": 64,
            "epochs": 60,
            "fc_dropout": 0.50,
            "use_augmentation": True,
            "use_class_weight": True
        },
        {
            "name": "exp5_sigmoid_lr3e4_aug_classw",
            "activation": "sigmoid",
            "lr": 3e-4,
            "batch_size": 64,
            "epochs": 60,
            "fc_dropout": 0.50,
            "use_augmentation": True,
            "use_class_weight": True
        }
    ]

    all_results = []
    for config in experiments:
        result = train_one_experiment(
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=args.output_dir,
            class_weight=class_weight
        )
        all_results.append(result)

    best_result = max(all_results, key=lambda x: x["best_val_accuracy"])
    save_json(all_results, os.path.join(args.output_dir, "all_experiment_results.json"))
    save_json(best_result, os.path.join(args.output_dir, "best_experiment.json"))

    print("\n=========================================")
    print("Best experiment selected by val_accuracy")
    print(json.dumps(best_result, indent=2, ensure_ascii=False))
    print("=========================================\n")

    best_exp_dir = os.path.join(args.output_dir, best_result["name"])
    test_result = evaluate_best_model(
        model_path=best_result["checkpoint_path"],
        X_test=X_test,
        y_test=y_test,
        output_dir=best_exp_dir
    )

    print("Final test result:")
    print(json.dumps(test_result, indent=2, ensure_ascii=False))

    sample_index = 0
    visualize_feature_maps(
        model_path=best_result["checkpoint_path"],
        image=X_test[sample_index],
        true_label=int(y_test[sample_index]),
        save_path=os.path.join(best_exp_dir, "feature_maps_sample0.png"),
        layer_names=["block1_conv1", "block2_conv1", "block3_conv1"],
        max_maps=8
    )

    best_model = tf.keras.models.load_model(best_result["checkpoint_path"])
    probs = best_model.predict(X_test[:20], verbose=0)
    preds = np.argmax(probs, axis=1)

    pred_rows = []
    for i in range(20):
        pred_rows.append({
            "index": i,
            "true_label": int(y_test[i]),
            "true_name": CLASS_NAMES[int(y_test[i])],
            "pred_label": int(preds[i]),
            "pred_name": CLASS_NAMES[int(preds[i])],
            "confidence": float(np.max(probs[i]))
        })

    save_json(pred_rows, os.path.join(best_exp_dir, "sample_predictions.json"))

    print(f"\nAll outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()