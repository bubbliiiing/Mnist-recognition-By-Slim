import tensorflow as tf
import numpy as np

# 创建slim对象
slim = tf.contrib.slim

class Conv_Net(object):

    def net(self,inputs):
        with tf.variable_scope("Net"):
            # 第一个卷积层
            net = slim.conv2d(inputs,32,[5,5],padding = "SAME",scope = 'conv1_1')
            net = slim.max_pool2d(net,[2,2],stride = 2,padding = "SAME",scope = 'pool1')
            
            # 第二个卷积层
            net = slim.conv2d(net,64,[3,3],padding = "SAME",scope = 'conv2_1')
            net = slim.max_pool2d(net,[2,2],stride = 2,padding = "SAME",scope = 'pool2')

            net = tf.reshape(net,[-1,7*7*64])
            # 全连接层
            layer1 = slim.fully_connected(net,512,scope = 'fully1')
            layer1 = slim.dropout(layer1, 0.5, scope='dropout1')
            layer3 = slim.fully_connected(layer1,10,activation_fn=tf.nn.softmax,scope = 'fully3')

            return layer3

    def get_loss(self,labels,logits):
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits = logits),name = 'loss')
            tf.summary.scalar("loss",loss)
        return loss
