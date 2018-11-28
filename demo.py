import os

import tensorflow as tf
import cv2
import tensorflow.contrib.slim as slim
import codecs
import sys
import time
import random
sys.path.append('./')

from nets import txtbox_384, np_methods, txtbox_768
from processing import ssd_vgg_preprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '3' #using GPU 0

def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            xmin = int(bboxes[i, 0] * width)
            ymin = int(bboxes[i, 1] * height)
            xmax = int(bboxes[i, 2] * width)
            ymax = int(bboxes[i, 3] * height)
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0))
    return img

gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.3)

config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config) 

# Input placeholder.
net_shape = (384, 384)
#net_shape = (768, 768)
data_format = 'NHWC'
img_input = tf.placeholder(tf.float32, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img, xs, ys = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None,None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
image_4d = tf.cast(image_4d, tf.float32)
# Define the txt_box model.
reuse = True if 'txt_net' in locals() else None

txt_net = txtbox_384.TextboxNet()
print(txt_net.params.img_shape)
print('reuse:',reuse)

with slim.arg_scope(txt_net.arg_scope(data_format=data_format)):
    predictions,localisations, logits, end_points = txt_net.net(image_4d, is_training=False, reuse=reuse)

ckpt_dir = 'model'

isess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

ckpt_filename = tf.train.latest_checkpoint(ckpt_dir)
if ckpt_dir and ckpt_filename:
    print('checkpoint:',ckpt_dir, os.getcwd(), ckpt_filename)
    saver.restore(isess, ckpt_filename)
    txt_anchors = txt_net.anchors(net_shape)

    def process_image(img, select_threshold=0.01, nms_threshold=.45, net_shape=net_shape):
        # Run txt network.
        startTime = time.time()
        rimg, rpredictions,rlogits,rlocalisations, rbbox_img = isess.run([image_4d, predictions, logits, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        end_time = time.time()
        print(end_time - startTime)
        # Get classes and bboxes from the net outputs

        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, txt_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        # print(rscores)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes


    # Test on some demo image and visualize output.
    path = './demo/'

    img = cv2.imread(path + 'img_1.jpg')
    img_cp = img.copy()
    rclasses, rscores, rbboxes = process_image(img_cp)

    with codecs.open('demo/detections.txt', 'w', encoding='utf-8') as fout:
        for i in range(len(rclasses)):
            fout.write('{},{}\n'.format(rbboxes[i], rscores[i]))
    img_with_bbox = plt_bboxes(img_cp, rclasses, rscores, rbboxes)
    cv2.imwrite(os.path.join(path,'demo_res.png'), img_with_bbox)
    print('detection finished')
else:
    raise ('no ckpt')

