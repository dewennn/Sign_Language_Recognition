import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import cv2
import tensorflow as tf
import skimage

from scipy.cluster.vq import *

import pickle

# Import the necessary files with pickle
with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('Bag-of-Feature_NN_Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    std_scaler = pickle.load(f)


orb = cv2.ORB_create()
centroids = kmeans.cluster_centers_

def inverse_transform_fn(predictions):
  return encoder.inverse_transform(np.argmax(predictions, axis=-1).reshape(-1, 1))

def return_histogram(img):
  _, descriptor = orb.detectAndCompute(img, None)

  if descriptor is None:
    descriptor = np.zeros((1, orb.descriptorSize()), dtype=np.uint8)

  image_features = np.zeros(len(centroids), 'float32')

  words, _ = vq(descriptor, centroids)

  for w in words:
    image_features[w] += 1

  return image_features.reshape(1, -1)