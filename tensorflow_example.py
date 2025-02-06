import tensorflow as tf
from tensorflow.keras import layers,models


from tensorflow.keras.datasets import mnist
import numpy as np
(x_train,y_train),(x_test,y_test)=mnist.load_data()




x_train,x_test=x_train/255.0,y_train/255.0
