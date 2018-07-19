"""
This script sets up and trains a neural to map the inverse kinematics
of a tendon driven continuum robot consisting of 2 segments
"""
import numpy as np
#import quaternion
import tensorflow as tf
import keras
from keras.layers import Dense
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from forward_kinematics import TendonDrivenContinuumRobot
import matplotlib.pyplot as plt
import sys
from keras.callbacks import TensorBoard

n = 355000
filename = "/home/andi/Documents/kinematics_continuum_robot/data/data"+ str(n) +".pkl"
data = pickle.load(open(filename, "rb"))

x_pos = data["positions"]
x_orientation = data["orientations"]
x = np.concatenate((x_pos, x_orientation), axis=1)

y = data["lengths"]

# test_sizes are chosen so that, x_train = 70% of data
# x_val = 15% of data, x_test = 15% of data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(0.15/0.85), random_state=1)

# Preprocessing: Normalization to range [-1, 1]
# x = 2*(x-xmin)/(xmax-xmin)-1
#x_train_means = np.mean(x_train, axis=0)
x_train_mins = np.min(x_train, axis=0)
x_train_maxs = np.max(x_train, axis=0)
x_train_means = np.mean(x_train, axis=0)
x_train_stds = np.std(x_train, axis=0)

x_train_normalized = 2*(x_train-x_train_mins)/(x_train_maxs-x_train_mins) - 1
x_val_normalized = 2*(x_val-x_train_mins)/(x_train_maxs-x_train_mins) - 1
x_test_normalized = 2*(x_test-x_train_mins)/(x_train_maxs-x_train_mins) -1

x_train_scaled = (x_train-x_train_means)/x_train_stds
x_val_scaled = (x_val-x_train_means)/x_train_stds
x_test_scaled = (x_val-x_train_means)/x_train_stds

x_train_normalized_sklearn = sklearn.preprocessing.minmax_scale(x_train, feature_range=(-1, 1), axis=0)
#x_val_normalized_sklearn = sklearn.preprocessing.minmax_scale(x_val, feature_range=(-1, 1), axis=0)
#x_test_normalized_sklearn = sklearn.preprocessing.minmax_scale(x_test, feature_range=(-1, 1), axis=0)

print("x_train similarity is {}".format(np.allclose(x_train_normalized, x_train_normalized_sklearn)))
#print("x_val similarity is {}".format(np.allclose(x_val_normalized, x_val_normalized_sklearn)))
#print("x_test similarity is {}".format(np.allclose(x_test_normalized, x_test_normalized_sklearn)))

#sys.exit("HUHU")

#model = create_model([32, 32])
units = 64
model = keras.Sequential()
model.add(Dense(units, activation=keras.activations.relu, input_dim=7))
#model.add(Dense(units, activation=keras.activations.relu))
model.add(Dense(6, activation=None))

model.compile(
   optimizer=keras.optimizers.Adam(lr=0.0001),
   loss='MSE',
   metrics=[]
)

model.summary()

EarlyStoppingCallback = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1)
TensorboardCallback = TensorBoard()
#
history = model.fit(
      x_train,
      y_train,
      epochs=1000,
      batch_size=512,
      validation_data=(x_val_scaled, y_val),
      verbose=1,
      callbacks=[EarlyStoppingCallback, TensorboardCallback]
      )

mse = model.evaluate(x_test, y_test)

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, "bo", label="Training loss", markersize=1)
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
#plt.ylim((0, 5e-5))
plt.legend()
plt.show()

position = 3

test_x = (np.expand_dims(x_test[position], 0))

prediction_lengths = model.predict(test_x)
y_test_lengths = y_test[position]
print(prediction_lengths[0]); print(y_test_lengths)

#env = TendonDrivenContinuumRobot()
#
#print(env.forward_kinematics(tendon_lengths=prediction_lengths))
#print(env.forward_kinematics(tendon_lengths=y_test_lengths.reshape((1,6))))
