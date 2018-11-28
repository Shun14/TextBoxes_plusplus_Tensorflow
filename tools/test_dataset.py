from datasets import sythtextprovider
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from nets import txtbox_384
from processing import ssd_vgg_preprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO) 

show_pic_sum = 10
save_dir = 'pic_test_dataset'
tf.app.flags.DEFINE_string(
    'dataset_dir', 'tfrecord_train', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'num_readers', 2,
    'The number of parallel readers that read data from the dataset.')
 
tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

FLAGS = tf.app.flags.FLAGS
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def draw_polygon(img,x1,y1,x2,y2,x3,y3,x4,y4, color=(255, 0, 0)):
    # print(x1, x2, x3, x4, y1, y2, y3, y4)
    x1 = int(x1)
    x2 = int(x2)
    x3 = int(x3)
    x4 = int(x4)

    y1 = int(y1)
    y2 = int(y2)
    y3 = int(y3)
    y4 = int(y4)
    cv2.line(img,(x1,y1),(x2,y2),color,2)
    cv2.line(img,(x2,y2),(x3,y3),color,2)
    cv2.line(img,(x3,y3),(x4,y4),color,2)
    cv2.line(img,(x4,y4),(x1,y1),color,2)
    # cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('test.png', img)
    return img


def run():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    print('-----start test-------')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with tf.device('/GPU:0'):
        dataset = sythtextprovider.get_datasets(FLAGS.dataset_dir)
        print(dataset)
        provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers=FLAGS.num_readers,
                        common_queue_capacity=20 * FLAGS.batch_size,
                        common_queue_min=10 * FLAGS.batch_size,
                        shuffle=True)
        print('provider:',provider)
        [image, shape, glabels, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get(['image', 'shape',
                                                                 'object/label',
                                                                 'object/bbox',
                                                                 'object/oriented_bbox/x1',
                                                                 'object/oriented_bbox/x2',
                                                                 'object/oriented_bbox/x3',
                                                                 'object/oriented_bbox/x4',
                                                                 'object/oriented_bbox/y1',
                                                                 'object/oriented_bbox/y2',
                                                                 'object/oriented_bbox/y3',
                                                                 'object/oriented_bbox/y4'
                                                                 ])
        print('image:',image)
        print('shape:',shape)
        print('glabel:',glabels)
        print('gboxes:',gbboxes)
        
        
        gxs = tf.transpose(tf.stack([x1,x2,x3,x4])) #shape = (N,4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        
        image = tf.identity(image, 'input_image')
        text_shape = (384, 384)
        image, glabels, gbboxes, gxs, gys= ssd_vgg_preprocessing.preprocess_image(image,  glabels,gbboxes,gxs, gys,
                                                            text_shape,is_training=True,
                                                            data_format='NHWC')
        
        
        x1, x2 , x3, x4 = tf.unstack(gxs, axis=1)
        y1, y2, y3, y4 = tf.unstack(gys, axis=1)
        
        text_net = txtbox_384.TextboxNet()
        text_anchors = text_net.anchors(text_shape)
        e_localisations, e_scores, e_labels = text_net.bboxes_encode( glabels, gbboxes, text_anchors, gxs, gys)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        j = 0
        all_time = 0
        try:
            while not coord.should_stop() and j < show_pic_sum:
                start_time = time.time()
                image_sess, label_sess, gbbox_sess,  x1_sess, x2_sess, x3_sess, x4_sess, y1_sess, y2_sess, y3_sess, y4_sess,p_localisations, p_scores, p_labels = sess.run([
                    image, glabels, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4,e_localisations , e_scores, e_labels])
                end_time = time.time() - start_time
                all_time += end_time
                image_np = image_sess
                # print(image_np)
                # print('label_sess:',label_sess)
                
                p_labels_concat = np.concatenate(p_labels)
                p_scores_concat = np.concatenate(p_scores)
                debug = False
                if debug is True:
                    print(p_labels)
                    print('l_labels:',  len(p_labels_concat[p_labels_concat.nonzero()]),p_labels_concat[p_labels_concat.nonzero()] )
                    print('p_socres:', len(p_scores_concat[p_scores_concat.nonzero()]), p_scores_concat[p_scores_concat.nonzero()])
                    # print(img_np.shape)

                    print('label_sess:', np.array(list(label_sess)).shape, list(label_sess))
                img_np = np.array(image_np)
                cv2.imwrite('{}/{}.png'.format(save_dir, j), img_np)
                img_np = cv2.imread('{}/{}.png'.format(save_dir, j))
                
                h, w, d = img_np.shape

                label_sess = list(label_sess)
                # for i , label in enumerate(label_sess):
                i = 0
                num_correct = 0
                
                for label in label_sess:
                    # print(int(label) == 1)
                    if int(label) == 1:
                        num_correct += 1
                        img_np = draw_polygon(img_np,x1_sess[i] * w, y1_sess[i]*h, x2_sess[i]*w, y2_sess[i]*h, x3_sess[i]*w, y3_sess[i]*h, x4_sess[i]*w, y4_sess[i]*h)
                    if int(label) == 0:
                        img_np = draw_polygon(img_np,x1_sess[i] * w, y1_sess[i]*h, x2_sess[i]*w, y2_sess[i]*h, x3_sess[i]*w, y3_sess[i]*h, x4_sess[i]*w, y4_sess[i]*h, color=(0, 0, 255))
                    i += 1
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                cv2.imwrite('{}'.format(os.path.join(save_dir, str(j)+'.png')), img_np)
                j+= 1
                print('correct:', num_correct)
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            print('done')
            coord.request_stop()
        print('all time:', all_time, 'average:', all_time / show_pic_sum)
        coord.join(threads=threads)

if __name__ == '__main__':
    run()
