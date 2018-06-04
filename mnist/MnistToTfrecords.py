# -*- coding: utf-8 -*-
"""
将 mnist 保存成 tfrecord
"""
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist


# 读取 mnist 数据，转为 头发record格式
def write_mnist_to_tfrecord(save_dir = "MNIST_data"):
    '''
    mnist
    :return: ['height', 'width', 'depth', 'label', 'image_raw']
    '''
    data_sets = mnist.read_data_sets(save_dir, dtype=tf.uint8, reshape=False, validation_size=1000)

    data_splits = ["train", "test", "validation"]
    for d in range(len(data_splits)):
        print("saving " + data_splits[d])
        data_set = data_sets[d]

        filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(data_set.images.shape[0]):
            # 转为字符串
            image = data_set.images[index].tostring()
            # 每个记录的格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_set.labels[index])])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
            # 写入
            writer.write(example.SerializeToString())
        writer.close()
        pass
    pass


# 从 tfrecord 中读取mnist数据
def read_mnist_from_tfrecord(save_dir = "MNIST_data", data_type='train', num_epochs = 1):
    ''' 
    :return: [image, label]
    '''
    filename = os.path.join(save_dir, "{}.tfrecords".format(data_type))
    # 将每个文件放入队列中 num_epochs次
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    # tfrord文件读取器
    reader = tf.TFRecordReader()
    # 读取单个实例
    _, serialized_example = reader.read(filename_queue)
    # 解析实例
    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64),
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])

    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    # Shuffle the examples + batch
    # 读入 一定数目内容放入新的队列中
    return tf.train.shuffle_batch([image, label], batch_size=2, capacity=2000, min_after_dequeue=1000)
    pass


if __name__ == '__main__':

    images_batch, labels_batch = read_mnist_from_tfrecord()

    # 可以忽略
    W = tf.get_variable("W", [28 * 28, 10])
    y_pred = tf.matmul(images_batch, W)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
    loss_mean = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                step += 1
                sess.run([train_op])
                if step % 500 == 0:
                    loss_mean_val = sess.run([loss_mean])
                    print(step)
                    print(loss_mean_val)
        except tf.errors.OutOfRangeError:
            print('{} steps.'.format(step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish
    coord.join(threads)
    sess.close()
    pass
