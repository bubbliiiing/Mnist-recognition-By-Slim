import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 生成整数的属性
def _int64_feature(value):
    if not isinstance(value,list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 生成浮点数的属性
def _float_feature(value):
    if not isinstance(value,list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# 生成字符串型的属性
def _bytes_feature(value):
    if not isinstance(value,list) and not isinstance(value,np.ndarray):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# 读取MNIST数据集
mnist = input_data.read_data_sets('./MNIST_data', dtype=tf.float32, one_hot=True)

# 获得image，shape为(55000,784)
images = mnist.train.images
# 获得label，shape为(55000,10)
labels = mnist.train.labels
# 获得一共具有多少张图片
num_examples = mnist.train.num_examples

# 存储TFRecord文件的地址
filename = 'record/output.tfrecords'
# 创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)

# 将每张图片都转为一个Example，并写入
for i in range(num_examples):
    image_raw = images[i]  # 读取每一幅图像
    image_string = images[i].tostring()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/class/label': _int64_feature(np.argmax(labels[i])),
                'image/encoded': _float_feature(image_raw),
                'image/encoded_tostring': _bytes_feature(image_string)
            }
        )
    )
    print(i,"/",num_examples)
    writer.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('data processing success')
writer.close()