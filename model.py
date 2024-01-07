import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0
model = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scale, y_train)

def getPrediction(image):
  imgpil = Image.open(image)
  img_bw = imgpil.convert('L')
  img_bw_resized = img_bw.resize((28, 28), Image.ANTIALIAS)
  pixel_filter = 20
  min_pixel = np.percentile(img_bw_resized, pixel_filter)
  img_bw_resized_scaled = np.clip(img_bw_resized - min_pixel, 0, 255)
  max_pixel = np.max(img_bw_resized)
  img_bw_resized_scaled = np.asarray(img_bw_resized_scaled)/max_pixel
  test_sample = np.array(img_bw_resized_scaled).reshape(1, 784)
  test_prediction = model.predict(test_sample)
  return test_prediction[0]
