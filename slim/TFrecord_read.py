import tensorflow as tf
import numpy as np

# 创建一个reader来读取TFRecord文件中的Example
reader = tf.TFRecordReader()

# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['record/output.tfrecords'])

# 从文件中读出一个Example
_, serialized_example = reader.read(filename_queue)

# 用parse_single_example将读入的Example解析成tensor
features = tf.parse_single_example(
    serialized_example,
    features={
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/encoded': tf.FixedLenFeature([784], tf.float32, default_value=tf.zeros([784], dtype=tf.float32)),
        'image/encoded_tostring': tf.FixedLenFeature([], tf.string)
    }
)

# 将字符串解析成图像对应的像素数组
labels = tf.cast(features['image/class/label'], tf.int32)
images = tf.cast(features['image/encoded'], tf.float32)
images_tostrings = tf.decode_raw(features['image/encoded_tostring'], tf.float32)

sess = tf.Session()

# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行读取一个Example。当所有样例读取完之后，在此样例中程序会重头读取
for i in range(5):
    label, image = sess.run([labels, images])
    images_tostring = sess.run(images_tostrings)
    print(np.shape(image))
    print(np.shape(images_tostring))
    print(label)
    print("#########################")