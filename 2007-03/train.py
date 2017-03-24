from __future__ import print_function
import tensorflow as tf
import sklearn
import sklearn.datasets
import pandas as pd
import requests
import numpy as np
import io
from tensorflow.contrib import legacy_seq2seq
from tensorflow.python.ops.math_ops import sigmoid, tanh

class Model(object):
  def __init__(self):
    input_size = 2
    hidden_size = 3
    output_size = 2
    learning_rate = 1

    self.x = tf.placeholder(tf.float32, shape=[None, 2])
    self.y = tf.placeholder(tf.float32, shape=[None, 2])
    Wh = tf.Variable(tf.random_normal([input_size, hidden_size]))
    Wo = tf.Variable(tf.random_normal([hidden_size, output_size]))
    bh = tf.Variable(tf.zeros([hidden_size]))
    bo = tf.Variable(tf.zeros([output_size]))
    self.weights = [Wh, Wo, bh, bo]

    self.h = sigmoid(tf.matmul(self.x, Wh) + bh)
    self.o = tf.nn.softmax(tf.matmul(self.h, Wo) + bo)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(self.o, 1), tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.o, labels=self.y))
    self.op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

def train(X=None, T=None):
  training_epochs = 3000

  model = Model()

  if X is None or T is None:
    # Generate input data
    num_samples = 100
    X, t = sklearn.datasets.make_circles(n_samples=num_samples, shuffle=False, factor=0.3, noise=0.1)
    T = np.zeros((100, 2)) # Define target matrix
    T[t == 1, 1] = 1
    T[t == 0, 0] = 1

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
      res = sess.run([model.op, model.accuracy, model.loss], feed_dict={model.x: X, model.y: T})

      if epoch % 10 == 0:
        print("Epoch: ", epoch, "Accuracy: ", res[1], res[2])

      if res[1] >= 1:
        break

    print("Accuracy: ", model.accuracy.eval(feed_dict={model.x: X, model.y: T}))
    print("done")

    res = sess.run(model.weights, feed_dict={model.x: X, model.y: T})

  return res

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
