# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a SSD model using a given dataset."""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import sythtextprovider
from deployment import model_deploy
from nets import txtbox_384, txtbox_768
from processing import ssd_vgg_preprocessing
from tf_extended import tf_utils
import os
import tensorflow.contrib.slim as slim

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# =========================================================================== #
# Text Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float('loss_alpha', 0.2,
                          'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float('negative_ratio', 3.,
                          'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('match_threshold', 0.5,
                          'Matching threshold in the loss function.')
tf.app.flags.DEFINE_boolean('large_training', False, 'Use 768 to train')
# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir',
    'icdar15_model/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10,
                            'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 1200,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.9,
                          'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1,
                          'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0,
                          'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1,
                          'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0,
                          'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0,
                          'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          'The amount of label smoothing.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 40000,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None, 'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('dataset_name', 'sythtext',
                           'The name of the dataset to load.')
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string('dataset_split_name', 'train',
                           'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './icdar15_tf',
    ' The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string('model_name', 'text_box_384',
                           'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None,
    'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_string('training_image_crop_area', '0.1, 1.0',
                           'the area of image process for training')
tf.app.flags.DEFINE_integer('max_number_of_steps', 60000,
                            'The maxim number of training steps.')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    #'checkpoint_path','/home/zsz/code/TextBoxes_plusplus_Tensorflow/vgg_model/vgg_16.ckpt',
    'checkpoint_path', './polygon_model/',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # Select the dataset.

        dataset = sythtextprovider.get_datasets(FLAGS.dataset_dir)
        # Get the TextBoxes++ network and its anchors.
        text_net = txtbox_384.TextboxNet()
        if FLAGS.large_training:
            text_net.params = text_net.params._replace(img_shape = (768, 768))
            text_net.params = text_net.params._replace(feat_shapes = [(96, 96), (48,48), (24, 24), (12, 12), (10, 10), (8, 8)])
        text_shape = text_net.params.img_shape
        print('text_shape ' + str(text_shape))
        text_anchors = text_net.anchors(text_shape)

        tf_utils.print_configuration(FLAGS.__flags, text_net.params,
                                     dataset.data_sources, FLAGS.train_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #

        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=1000 * FLAGS.batch_size,
                    common_queue_min=300 * FLAGS.batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes, x1, x2, x3, x4, y1, y2, y3,
             y4] = provider.get([
                 'image', 'shape', 'object/label', 'object/bbox',
                 'object/oriented_bbox/x1', 'object/oriented_bbox/x2',
                 'object/oriented_bbox/x3', 'object/oriented_bbox/x4',
                 'object/oriented_bbox/y1', 'object/oriented_bbox/y2',
                 'object/oriented_bbox/y3', 'object/oriented_bbox/y4'
             ])
            gxs = tf.transpose(tf.stack([x1, x2, x3, x4]))  #shape = (N,4)
            gys = tf.transpose(tf.stack([y1, y2, y3, y4]))

            image = tf.identity(image, 'input_image')

            init_op = tf.global_variables_initializer()
            # tf.global_variables_initializer()
            # Pre-processing image, labels and bboxes.
            training_image_crop_area = FLAGS.training_image_crop_area
            area_split = training_image_crop_area.split(',')
            assert len(area_split) == 2
            training_image_crop_area = [
                float(area_split[0]),
                float(area_split[1])
            ]

            image, glabels, gbboxes, gxs, gys= \
                ssd_vgg_preprocessing.preprocess_for_train(image,  glabels,gbboxes,gxs, gys,
                                                        text_shape,
                                                        data_format='NHWC', crop_area_range=training_image_crop_area)

            # Encode groundtruth labels and bboxes.

            image = tf.identity(image, 'processed_image')

            glocalisations, gscores, glabels = \
                text_net.bboxes_encode( glabels, gbboxes, text_anchors, gxs, gys)
            batch_shape = [1] + [len(text_anchors)] * 3

            # Training batches and queue.

            r = tf.train.batch(
                tf_utils.reshape_list(
                    [image, glocalisations, gscores, glabels]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

            b_image, b_glocalisations, b_gscores, b_glabels= \
                tf_utils.reshape_list(r, batch_shape)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            batch_queue = slim.prefetch_queue.prefetch_queue(
                tf_utils.reshape_list(
                    [b_image, b_glocalisations, b_gscores, b_glabels]),
                capacity=2 * deploy_config.num_clones)

        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        def clone_fn(batch_queue):

            #Allows data parallelism by creating multiple
            #clones of network_fn.
            # Dequeue batch.
            b_image, b_glocalisations, b_gscores, b_glabels = \
                tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

            # Construct TextBoxes network.
            arg_scope = text_net.arg_scope(weight_decay=FLAGS.weight_decay)
            with slim.arg_scope(arg_scope):
                predictions,localisations, logits, end_points = \
                    text_net.net(b_image, is_training=True)
            # Add loss function.

            text_net.losses(
                logits,
                localisations,
                b_glabels,
                b_glocalisations,
                b_gscores,
                match_threshold=FLAGS.match_threshold,
                negative_ratio=FLAGS.negative_ratio,
                alpha=FLAGS.loss_alpha,
                label_smoothing=FLAGS.label_smoothing,
                batch_size=FLAGS.batch_size)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # =================================================================== #
        # Add summaries from first clone.
        # =================================================================== #
        clones = model_deploy.create_clones(deploy_config, clone_fn,
                                            [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(
                tf.summary.scalar('sparsity/' + end_point,
                                  tf.nn.zero_fraction(x)))
        # Add summaries for losses and extra losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        for loss in tf.get_collection('EXTRA_LOSSES'):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(
                FLAGS, dataset.num_samples, global_step)
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(
                variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = tf_utils.get_variables_to_train(FLAGS)

        # and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones, optimizer, var_list=variables_to_train)
        # Add total_loss to summary.

        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(
            clones_gradients, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies(
            [update_op], total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

        config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options)
        saver = tf.train.Saver(
            max_to_keep=100,
            keep_checkpoint_every_n_hours=1.0,
            write_version=2,
            pad_step_number=False)

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            master='',
            is_chief=True,
            # init_op=init_op,
            init_fn=tf_utils.get_init_fn(FLAGS),
            summary_op=summary_op,  ##output variables to logdir
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=config,
            sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()
