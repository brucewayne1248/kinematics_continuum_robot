import numpy as np
import tensorflow as tf
import tensorboard
import quaternion
import pickle, scipy.io
import sklearn
from sklearn.model_selection import train_test_split

n = 355000

filename = "/home/andi/Documents/kinematics_continuum_robot/data/data" + str(n) + ".pkl"
data = pickle.load( open(filename, "rb") )

x_pos = data["positions"]
x_orientation = data["orientations"]
x = np.concatenate((x_pos, x_orientation), axis=1)
y = data["lengths"]

########## PREPROCESSING ##########
# test_sizes are chosen so that, x_train = 70% of data
# x_val = 15% of data, x_test = 15% of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(0.15/0.85), random_state=1)
# Preprocessing: Normalization to range [-1, 1]
# x = 2*(x-xmin)/(xmax-xmin)-1
x_train_mins = np.min(x_train, axis=0)
x_train_maxs = np.max(x_train, axis=0)
x_train_normalized = 2*(x_train-x_train_mins)/(x_train_maxs-x_train_mins) - 1
x_val_normalized = 2*(x_val-x_train_mins)/(x_train_maxs-x_train_mins) - 1
x_test_normalized = 2*(x_test-x_train_mins)/(x_train_maxs-x_train_mins) -1
x_train_normalized_sklearn = sklearn.preprocessing.minmax_scale(x_train, feature_range=(-1, 1), axis=0)
#print("x_train similarity is {}".format(np.allclose(x_train_normalized, x_train_normalized_sklearn)))
# Preprocessing: x to have zero mean and std = 1
x_train_means = np.mean(x_train, axis=0)
x_train_stds = np.std(x_train, acis=0)
x_train_scaled = (x_train-x_train_means)/x_train_stds
x_val_scaled = (x_val-x_train_means)/x_train_stds
x_test_scaled = (x_test-x_train_means)/x_train_stds


tf.reset_default_graph()
dim_input = 7
dim_output = 6

def create_model(dims, activation="relu", initializer="glorot"):
   input_dim = dims[0]; hidden_units = dims[1:-1]; output_dim = dims[-1]




with tf.name_scope("Model"):
   tf_x = tf.placeholder(tf.float32, [None, dim_input], name="input")
   tf_y = tf.placeholder(tf.float32, [None, dim_output], name="output")

   W1 = tf.get_variable("W1", shape=)


with tf.name_scope("Loss"):
   pass

with tf.name_scope("Optimizer"):
   pass

with tf.name_scope("fit_model"):
   pass

with tf.Session() as sess:
   pass


train_network()



