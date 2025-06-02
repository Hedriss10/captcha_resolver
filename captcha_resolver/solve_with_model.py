import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
import time

IMG_WIDTH, IMG_HEIGHT = 200, 50
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

model = tf.keras.models.load_model("model/captcha_model.h5")

def predict_captcha_from_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, h, w, 1)

    preds = model.predict(img)
    result = ''.join([CHARS[np.argmax(p)] for p in preds])
    return result

def solve_captcha(driver):
    # captcha_element = driver.find_element(By.XPATH, "//img[@name='captcha_img']")
    # png = captcha_element.screenshot_as_png
    png = ...
    img = Image.open(BytesIO(png))
    img_np = np.array(img)

    predicted_text = predict_captcha_from_image(img_np)
    print("Captcha predicted:", predicted_text)

    # campo = driver.find_element(By.NAME, "captcha")
    # campo.clear()
    # campo.send_keys(predicted_text)
