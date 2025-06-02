import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# CONFIG
IMG_WIDTH, IMG_HEIGHT = 200, 50
MAX_LABEL_LEN = 5
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_DICT = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if not file.endswith(".png"): continue
        label = file.split(".")[0].lower()
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        X.append(img)
        # Encode label
        label_encoded = [CHAR_DICT.get(c, 0) for c in label]
        y.append(label_encoded)
    return np.array(X)[..., np.newaxis], np.array(y)

def pad_labels(y):
    return tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

def build_model():
    input_img = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = [
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name=f"char_{i}")
        for i in range(MAX_LABEL_LEN)
    ]

    output_layers = [layer(x) for layer in outputs]

    model = tf.keras.Model(inputs=input_img, outputs=output_layers)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    X, y_raw = load_data("data/labeled_images")
    X_train, X_val, y_train, y_val = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    y_train_split = [pad_labels(y_train[:, i]) for i in range(MAX_LABEL_LEN)]
    y_val_split = [pad_labels(y_val[:, i]) for i in range(MAX_LABEL_LEN)]

    model = build_model()
    model.fit(
        X_train,
        y_train_split,
        validation_data=(X_val, y_val_split),
        batch_size=32,
        epochs=15,
    )
    model.save("model/captcha_model.h5")
