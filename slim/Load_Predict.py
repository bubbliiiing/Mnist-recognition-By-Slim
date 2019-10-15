import tensorflow as tf
import numpy as np
from nets import Net
from tensorflow.examples.tutorials.mnist import input_data

def compute_accuracy(x_data,y_data):
    global prediction
    y_pre = sess.run(prediction,feed_dict={img_input:x_data})
    
    correct_prediction = tf.equal(tf.arg_max(y_data,1),tf.arg_max(y_pre,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    result = sess.run(accuracy,feed_dict = {img_input:x_data})
    return result

mnist = input_data.read_data_sets("MNIST_data",one_hot = "true")

slim = tf.contrib.slim

# img_input的placeholder
img_input = tf.placeholder(tf.float32, shape = (None, 784))
img_reshape = tf.reshape(img_input,shape = (-1,28,28,1))

# 载入模型
sess = tf.Session()

Conv_Net = Net.Conv_Net()

prediction = Conv_Net.net(img_reshape)

# 载入模型
ckpt_filename = './logs/model.ckpt-20000'

# 初始化所有变量
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# 恢复
saver.restore(sess, ckpt_filename)

print(compute_accuracy(mnist.test.images,mnist.test.labels))
