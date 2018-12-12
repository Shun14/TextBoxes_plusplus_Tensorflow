## an initial version
## Transform the tfrecord to slim data provider format

import numpy 
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import glob



ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
SPLITS_TO_SIZES = {
    #'train': 2518, for ppt datasets
    'train': 858750 # for synth text datasets
}
def get_datasets(data_dir,file_pattern = '*.tfrecord'):
    file_patterns = os.path.join(data_dir, file_pattern)
    print('file_path: {}'.format(file_patterns))
    file_path_list = glob.glob(file_patterns)
    num_samples = 0
    #num_samples = 288688
    #num_samples = 858750 only for synth datasets
    for file_path in file_path_list:
        for record in tf.python_io.tf_record_iterator(file_path):
            num_samples += 1
    print('num_samples:', num_samples) 
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y4': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ignored': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/oriented_bbox/x1': slim.tfexample_decoder.Tensor('image/object/bbox/x1'),
        'object/oriented_bbox/x2': slim.tfexample_decoder.Tensor('image/object/bbox/x2'),
        'object/oriented_bbox/x3': slim.tfexample_decoder.Tensor('image/object/bbox/x3'),
        'object/oriented_bbox/x4': slim.tfexample_decoder.Tensor('image/object/bbox/x4'),
        'object/oriented_bbox/y1': slim.tfexample_decoder.Tensor('image/object/bbox/y1'),
        'object/oriented_bbox/y2': slim.tfexample_decoder.Tensor('image/object/bbox/y2'),
        'object/oriented_bbox/y3': slim.tfexample_decoder.Tensor('image/object/bbox/y3'),
        'object/oriented_bbox/y4': slim.tfexample_decoder.Tensor('image/object/bbox/y4'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/ignored': slim.tfexample_decoder.Tensor('image/object/bbox/ignored')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {0:'background', 1:'text'}
    return slim.dataset.Dataset(
        data_sources=file_patterns,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names)

NUM_CLASSES = 2
