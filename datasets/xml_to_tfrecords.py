# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
import tensorflow.contrib.slim as slim

# TFRecords convertion parameters.

TXT_LABELS = {
	'none': (0, 'Background'),
	'text': (1, 'Text')
}

def _process_image(train_img_path, train_xml_path, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.

    image_data = tf.gfile.FastGFile(train_img_path, 'rb').read()

    tree = ET.parse(train_xml_path)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)
    if height <= 0 or width <= 0 or depth <= 0:
        print('height or width depth error',height, width, depth)
        return

    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    oriented_bbox = []
    ignored = 0
    filename = root.find('filename').text

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label == 'none':
            label = 'none'
        else:
            label = 'text'
        labels.append(int(TXT_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult') is not None:
            #print('append difficult')
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)
        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)

        x1 = float(bbox.find('x1').text)
        x2 = float(bbox.find('x2').text)
        x3 = float(bbox.find('x3').text)
        x4 = float(bbox.find('x4').text)

        y1 = float(bbox.find('y1').text)
        y2 = float(bbox.find('y2').text)
        y3 = float(bbox.find('y3').text)
        y4 = float(bbox.find('y4').text)


        ymin, ymax = np.clip([ymin, ymax], 0 , height)
        xmin, xmax = np.clip([xmin, xmax] ,0 , width)

        x1, x2, x3, x4 = np.clip([x1, x2, x3, x4], 0, width)
        y1, y2, y3, y4 = np.clip([y1, y2, y3, y4], 0, height)

        bboxes.append(( ymin / shape[0],
                        xmin / shape[1],
                        ymax / shape[0],
                        xmax / shape[1]
                       ))

        oriented_bbox.append((x1 / width, x2 / width, x3 / width, x4 /width, y1 /height, y2 / height, y3 / height, y4 / height))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated, oriented_bbox, ignored, filename

def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated, oriented_bbox, ignored, filename):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [ymin, xmin, ymax, xmax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    x1 = []
    x2 = []
    x3 = []
    x4 = []

    y1 = []
    y2 = []
    y3 = []
    y4 = []

    for orgin in oriented_bbox:
        assert  len(orgin) == 8
        [l.append(point) for l, point in zip([x1, x2, x3, x4, y1, y2, y3, y4], orgin)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/filename': bytes_feature(filename.encode('utf-8')),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/x1': float_feature(x1),
            'image/object/bbox/y1': float_feature(y1),
            'image/object/bbox/x2': float_feature(x2),
            'image/object/bbox/y2': float_feature(y2),
            'image/object/bbox/x3': float_feature(x3),
            'image/object/bbox/y3': float_feature(y3),
            'image/object/bbox/x4': float_feature(x4),
            'image/object/bbox/y4': float_feature(y4),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/object/bbox/ignored': int64_feature(ignored),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def _add_to_tfrecord(train_img_path, train_xml_path , name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      train_img_path: img path;
      train_xml_path: xml path
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated, oriented_bbox, ignored, filename = \
        _process_image(train_img_path, train_xml_path , name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated, oriented_bbox, ignored, filename)
    tfrecord_writer.write(example.SerializeToString())


def run(xml_img_txt_path, output_dir, name='icdar15_annotated_data', samples_per_files=200):
    """Runs the conversion operation.
	Args:
	  xml_img_txt_path: The txt stored where the dataset is stored.
	  output_dir: Output directory.
	"""

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    train_txt = open(xml_img_txt_path, 'r')
    lines = train_txt.readlines()
    train_img_path = []
    train_xml_path = []
    error_list = []
    count = 0
    for line in lines:
        line = line.strip()
        if len(line.split(',')) == 2:
            count += 1
            train_img_path.append(line.split(',')[0])
            train_xml_path.append(line.split(',')[1])
        else:
            error_list.append(line)
            # print('line:',line)
    with open(os.path.join(output_dir, 'create_tfrecord_error_list.txt'), 'w') as f:
        f.writelines(error_list)
    filenames = train_img_path

    # Process dataset files
    i = 0
    fidx = 0
    while i < len(filenames):
        #Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < samples_per_files:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename.split('/')[-1][:-4]
                _add_to_tfrecord(filename, train_xml_path[i], img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the charts dataset!')

