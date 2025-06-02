import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
import time

IMG_WIDTH, IMG_HEIGHT = 200, 50
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

path_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model/captcha_model_finetuned.h5")
path_img = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input/captcha_testing.png")

model = tf.keras.models.load_model("model/captcha_model.h5")


def predict_captcha_from_image(img_tensor):
    preds = model.predict(img_tensor)
    result = ''.join([CHARS[np.argmax(p)] for p in preds])
    return result

def preprocess_captcha_image(img):
    # Converte RGB → escala de cinza
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inversão: fundo preto, texto branco
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Redimensiona
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Normaliza e expande dimensão
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, h, w, 1)
    return img

def solve_captcha(path_img):
    with open(path_img, "rb") as f:
        png = f.read()
        img = Image.open(BytesIO(png)).convert("RGB")  # Garante 3 canais
        img_np = np.array(img)
        img_tensor = preprocess_captcha_image(img_np)
        predicted_text = predict_captcha_from_image(img_tensor)
        print("Captcha predicted:", predicted_text)

if __name__ == "__main__":
    solve_captcha(path_img=path_img)