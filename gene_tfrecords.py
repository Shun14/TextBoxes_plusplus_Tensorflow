# -*- coding: utf-8 -*-
import tensorflow as tf

from datasets import xml_to_tfrecords
import os
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'output_name', 'annotated_data',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', 'tfrecords',
    'Output directory where to store TFRecords files.')
tf.app.flags.DEFINE_string(
    'xml_img_txt_path', None,
    'the path means the txt'
)

tf.app.flags.DEFINE_integer(
    'samples_per_files', 2000,
    'the number means one tf_record save how many pictures'
)

def  main(_):
    if not FLAGS.xml_img_txt_path or not os.path.exists(FLAGS.xml_img_txt_path):
        raise ValueError('You must supply the dataset directory with --xml_img_txt_path')
    print('Dataset directory:', FLAGS.xml_img_txt_path)
    print('Output directory:', FLAGS.output_dir)

    xml_to_tfrecords.run(FLAGS.xml_img_txt_path, FLAGS.output_dir, FLAGS.output_name, FLAGS.samples_per_files)

if __name__ == '__main__':
    tf.app.run()
