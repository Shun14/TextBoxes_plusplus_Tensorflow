#encoding=utf-8
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim
import util



def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))


def convert_to_example(image_data, filename, labels, ignored, labels_text, bboxes, oriented_bboxes, shape):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image
      labels: list of integers, identifier for the ground truth
      labels_text: list of strings, human-readable labels
      oriented_bboxes: list of bounding oriented boxes each box is a list of floats in [0, 1]
          specifying [x1, y1, x2, y2, x3, y3, x4, y4]
      bboxes: list of bbox in rectangle, [xmin, ymin, xmax, ymax] 
    Returns:
      Example proto
    """
    
    image_format = b'JPEG'
    oriented_bboxes = np.asarray(oriented_bboxes)
    bboxes = np.asarray(bboxes)
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/shape': int64_feature(list(shape)),
            'image/object/bbox/xmin': float_feature(list(bboxes[:, 0])),
            'image/object/bbox/ymin': float_feature(list(bboxes[:, 1])),
            'image/object/bbox/xmax': float_feature(list(bboxes[:, 2])),
            'image/object/bbox/ymax': float_feature(list(bboxes[:, 3])),
            'image/object/bbox/x1': float_feature(list(oriented_bboxes[:, 0])),
            'image/object/bbox/y1': float_feature(list(oriented_bboxes[:, 1])),
            'image/object/bbox/x2': float_feature(list(oriented_bboxes[:, 2])),
            'image/object/bbox/y2': float_feature(list(oriented_bboxes[:, 3])),
            'image/object/bbox/x3': float_feature(list(oriented_bboxes[:, 4])),
            'image/object/bbox/y3': float_feature(list(oriented_bboxes[:, 5])),
            'image/object/bbox/x4': float_feature(list(oriented_bboxes[:, 6])),
            'image/object/bbox/y4': float_feature(list(oriented_bboxes[:, 7])),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/ignored': int64_feature(ignored),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(filename),
            'image/encoded': bytes_feature(image_data)}))
    return example



def get_split(split_name, dataset_dir, file_pattern, num_samples, reader=None):
    dataset_dir = util.io.get_absolute_path(dataset_dir)
    
    if util.str.contains(file_pattern, '%'):
        file_pattern = util.io.join_path(dataset_dir, file_pattern % split_name)
    else:
        file_pattern = util.io.join_path(dataset_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
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
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=items_to_descriptions,
            num_classes=2,
            labels_to_names=labels_to_names)




class SynthTextDataFetcher():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()
        
    # @util.dec.print_calling    
    def _load_mat(self):
        data = util.io.load_mat(self.mat_path)
        self.image_paths = data['imnames'][0]
        self.image_bbox = data['wordBB'][0]
        self.txts = data['txt'][0]
        self.num_images =  len(self.image_paths)

    def get_image_path(self, idx):
        image_path = util.io.join_path(self.root_path, self.image_paths[idx][0])
        return image_path

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except: # error caused by dataset
            return 1


    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) ==2: # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))
             
        xys = boxes[:,:, word_idx]
        assert(np.shape(xys) ==(2, 4))
        return np.float32(xys)
    
    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        
        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)
        
        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True
        
        if w < 10 or h < 10:
            is_valid = False
            
        if w * h < 100:
            is_valid = False
        
        xys[0, :] = xys[0, :] / width
        xys[1, :] = xys[1, :] / height
        
        return is_valid, min_x / width, min_y /height, max_x / width, max_y / height, xys
        
    def get_txt(self, image_idx, word_idx):
        txts = self.txts[image_idx]
        clean_txts = []
        for txt in txts:
            clean_txts += txt.split()
        return str(clean_txts[word_idx])
        
        
    def fetch_record(self, image_idx):
        image_path = self.get_image_path(image_idx)
        if not (util.io.exists(image_path)):
            return None
        img = util.img.imread(image_path)
        h, w = img.shape[0:-1]
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        full_bboxes = []
        txts = []
        for word_idx in range(num_words):
            xys = self.get_word_bbox(image_idx, word_idx)       
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width = w, height = h)
            if not is_valid:
                continue
            rect_bboxes.append([min_x, min_y, max_x, max_y])
            xys = np.reshape(np.transpose(xys), -1)
            full_bboxes.append(xys)
            txt = self.get_txt(image_idx, word_idx)
            txts.append(txt)
        if len(rect_bboxes) == 0:
            return None
        
        return image_path, img, txts, rect_bboxes, full_bboxes
    
        

def cvt_to_tfrecords(output_path , data_path, gt_path, records_per_file = 30000):

    fetcher = SynthTextDataFetcher(root_path = data_path, mat_path = gt_path)
    fid = 0
    image_idx = -1
    while image_idx < fetcher.num_images:
        with tf.python_io.TFRecordWriter(output_path%(fid)) as tfrecord_writer:
            record_count = 0
            while record_count != records_per_file:
                image_idx += 1
                if image_idx >= fetcher.num_images:
                    break
                print("loading image %d/%d"%(image_idx + 1, fetcher.num_images))
                record = fetcher.fetch_record(image_idx)
                if record is None:
                    print('\nimage %d does not exist'%(image_idx + 1))
                    continue

                image_path, image, txts, rect_bboxes, oriented_bboxes = record
                labels = len(rect_bboxes) * [1]
                ignored = len(rect_bboxes) * [0]
                image_data = tf.gfile.FastGFile(image_path, 'r').read()
                
                shape = image.shape
                image_name = str(util.io.get_filename(image_path).split('.')[0])
                example = convert_to_example(image_data, image_name, labels, ignored, txts, rect_bboxes, oriented_bboxes, shape)
                tfrecord_writer.write(example.SerializeToString())
                record_count += 1
                
        fid += 1
                    
if __name__ == "__main__":
    mat_path = util.io.get_absolute_path('/share/SynthText/gt.mat')
    root_path = util.io.get_absolute_path('/share/SynthText/')
    output_dir = util.io.get_absolute_path('/home/zsz/datasets/synth-tf/')
    util.io.mkdir(output_dir)
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir,  'SynthText_%d.tfrecord'), data_path = root_path, gt_path = mat_path)
