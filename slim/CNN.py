import tensorflow as tf
import numpy as np
from nets import Net
flags = tf.app.flags

#############################################################
#   设置训练参数
#############################################################

# =========================================================================== #
# General Flags.
# =========================================================================== #
# train_dir用于保存训练后的模型和日志
tf.app.flags.DEFINE_string(
    'train_dir', './logs',
    'Directory where checkpoints and event logs are written to.')
# num_readers是在对数据集进行读取时所用的平行读取器个数
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
# 在进行训练batch的构建时，所用的线程数
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
# 每十步进行一次log输出，在窗口上
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 100,
    'The frequency with which logs are print.')
# 每150秒存储一次记录
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 150,
    'The frequency with which summaries are saved, in seconds.')
# 每150秒存储一次模型
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 150,
    'The frequency with which the model is saved, in seconds.')
# 可以使用的gpu内存数量
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.6, 'GPU memory fraction to use.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
# 学习率衰减的方式，有固定、指数衰减等
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
# 初始学习率
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# 结束时的学习率
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
# 学习率衰减因素
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
# adam参数
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
# 数据集目录
tf.app.flags.DEFINE_string(
    'dataset_dir', './record/output.tfrecords', 'The directory where the dataset files are stored.')
# 每一次训练batch的大小
tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')
# 最大训练次数
tf.app.flags.DEFINE_integer('max_number_of_steps', 20000,
                            'The maximum number of training steps.')

FLAGS = flags.FLAGS

def get_record_dataset(record_path,
                       reader=None, image_shape=[784], 
                       num_samples=55000, num_classes=10):

    if not reader:
        reader = tf.TFRecordReader
        
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([784], tf.float32, default_value=tf.zeros([784], dtype=tf.float32)),
        'image/class/label':tf.FixedLenFeature([1], tf.int64, 
                                    default_value=tf.zeros([1], dtype=tf.int64))}
        
    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image/encoded', shape = [784]),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    
    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)

if __name__ == "__main__":

    # 打印日志
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        
        # 最大世代
        MAX_EPOCH = 50000

        # 创建slim对象
        slim = tf.contrib.slim

        # 步数
        global_step = slim.create_global_step()
        #############################################################
        #   读取MNIST数据集
        #############################################################
        # 读取数据集
        dataset = get_record_dataset(FLAGS.dataset_dir,num_samples = 55000)

        # 创建provider
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers= FLAGS.num_readers,
                    common_queue_capacity=20*FLAGS.batch_size,
                    common_queue_min=10*FLAGS.batch_size,
                    shuffle=True)
        
        # 在提供商处获取image
        image, label = provider.get(['image', 'label'])

        # 每次提取100个手写体
        inputs, labels = tf.train.batch([image, label],
                                        batch_size=FLAGS.batch_size,
                                        allow_smaller_final_batch=True,
                                        num_threads=FLAGS.num_readers,
                                        capacity=FLAGS.batch_size*5)

        #############################################################
        #建立卷积神经网络，并将数据集的image通过神经网络，获得prediction。
        #############################################################
        inputs = tf.cast(inputs,tf.float32)
        inputs_reshape = tf.reshape(inputs,[-1,28,28,1])

        Conv_Net = Net.Conv_Net()
        logits = Conv_Net.net(inputs_reshape)

        #############################################################
        #   利用prediction和实际label获得loss。
        #############################################################
        # 获得损失值
        loss = Conv_Net.get_loss(labels = labels,logits = logits)

        decay_steps = int(dataset.num_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

        # 学习率指数下降
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=False,
                                          name='exponential_decay_learning_rate')
        #############################################################
        #   利用优化器完成梯度下降并保存模型。
        #############################################################
        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # 构建训练对象
        train_op = slim.learning.create_train_op(loss, optimizer,
                                                summarize_gradients=False)
        # gpu使用比率
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                        allow_growth = True)
        # 参数配置
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                gpu_options=gpu_options)     
        # 保存方式
        saver = tf.train.Saver(max_to_keep=5,
                            keep_checkpoint_every_n_hours=1.0,
                            write_version=2,
                            pad_step_number=False)
        # 托管训练
        slim.learning.train(
                train_op,
                logdir=FLAGS.train_dir,
                master='',
                is_chief=True,
                number_of_steps = FLAGS.max_number_of_steps,
                log_every_n_steps = FLAGS.log_every_n_steps,
                save_summaries_secs= FLAGS.save_summaries_secs,
                saver=saver,
                save_interval_secs = FLAGS.save_interval_secs,
                session_config=config,
                sync_optimizer=None)

