# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import  math



# =========================================================================== #
# TensorFlow implementation of Text Boxes encoding / decoding.
# =========================================================================== #

def tf_text_bboxes_encode_layer(glabels, bboxes,gxs , gys,
                                anchors_layer,
                               matching_threshold=0.1,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):

    """
    Encode groundtruth labels and bounding boxes using Textbox anchors from
    one layer.

    Arguments:
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      gxs,gys: Nx4 Tensor
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_localizations, target_scores): Target Tensors.
    # this is a binary problem, so target_score and tartget_labels are same.
    """
    # Anchors coordinates and volume.

    # yref, xref, href, wref = anchors_layer
    xmin, ymin, xmax, ymax = anchors_layer
    xref = (xmin + xmax) /2
    yref = (ymin + ymax) /2
    href = ymax - ymin
    wref = xmax - xmin
    # caffe 源码是对每个pred_bbox（预测的bbox）和gt 进行overlap计算
    print(yref.shape)
    print(href.shape)
    print(bboxes.shape)
    # glabels = tf.Print(glabels, [tf.shape(glabels)], message=' glabels shape is :')
    #,,2
    # ymin = yref - href / 2.
    # xmin = xref - wref / 2.
    # ymax = yref + href / 2.
    # xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # bboxes = tf.Print(bboxes, [tf.shape(bboxes), bboxes], message=' bboxes in encode shape:', summarize=20)
    # glabels = tf.Print(glabels, [tf.shape(glabels)], message='  glabels shape:')

    # xs = np.asarray(gxs, dtype=np.float32)
    # ys = np.asarray(gys, dtype=np.float32)
    # num_bboxes = xs.shape[0]


    # Initialize tensors...
    shape = (yref.shape[0])
    # all after the flatten
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    feat_x1 = tf.zeros(shape, dtype=dtype)
    feat_x2 = tf.zeros(shape, dtype=dtype)
    feat_x3 = tf.zeros(shape, dtype=dtype)
    feat_x4 = tf.zeros(shape, dtype=dtype)
    feat_y1 = tf.zeros(shape, dtype=dtype)
    feat_y2 = tf.zeros(shape, dtype=dtype)
    feat_y3 = tf.zeros(shape, dtype=dtype)
    feat_y4 = tf.zeros(shape, dtype=dtype)


    # feat_x1 =tf.zeros()

    def jaccard_with_anchors(bbox):
        """
        Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard


    def intersection_with_anchors(bbox):
        '''
        Compute intersection between score a box and the anchors.
        '''
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i,feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax,
                  feat_x1, feat_x2, feat_x3, feat_x4,
                  feat_y1, feat_y2, feat_y3, feat_y4):
        """Condition: check label index.
        """

        r = tf.less(i, tf.shape(glabels)[0])

        return r

    def body(i,feat_labels, feat_scores,feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_x1, feat_x2, feat_x3, feat_x4, feat_y1, feat_y2, feat_y3, feat_y4):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = glabels[i]
        bbox = bboxes[i]
        gx = gxs[i]
        gy = gys[i]
        # i = tf.Print(i, [i , tf.shape(glabels), tf.shape(bboxes), tf.shape(gxs), tf.shape(gys)], message='i is :')
        jaccard = jaccard_with_anchors(bbox)
        # jaccard = tf.Print(jaccard, [tf.shape(jaccard), tf.nn.top_k(jaccard, 100, sorted=True)[0]], message=' jaccard :', summarize=100)
        # feat_scores = tf.Print(feat_scores, [tf.shape(feat_scores),tf.count_nonzero(feat_scores), tf.nn.top_k(feat_scores, 100, sorted=True)[0]], message= ' feat_scores: ', summarize= 100)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)

        # i = tf.Print(i, [tf.shape(i), i], message= ' i is: ')
        # tf.Print(mask, [mask])
        mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        # mask = tf.logical_and(mask, feat_scores > -0.5)
        # mask = tf.logical_and(mask, label < 2)
        # mask = tf.Print(mask, [tf.shape(mask), mask[0]], message=' mask is :')
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)
        # bbox ymin xmin ymax xmax gxs gys
        # update all box
        # bbox = tf.Print(bbox, [tf.shape(bbox), bbox], message= ' bbox : ', summarize=20)
        # gx = tf.Print(gx, [gx], message=' gx: ', summarize=20)
        # gy = tf.Print(gy, [gy], message= ' gy: ', summarize=20)
        # fmask = tf.Print(fmask, [tf.shape(fmask), tf.count_nonzero(fmask), tf.nn.top_k(fmask, 100, sorted=True)[0]], message=' fmask :', summarize=100)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        feat_x1 = fmask * gx[0] + (1 - fmask) * feat_x1
        feat_x2 = fmask * gx[1] + (1 - fmask) * feat_x2
        feat_x3 = fmask * gx[2] + (1 - fmask) * feat_x3
        feat_x4 = fmask * gx[3] + (1 - fmask) * feat_x4

        feat_y1 = fmask * gy[0] + (1 - fmask) * feat_y1
        feat_y2 = fmask * gy[1] + (1 - fmask) * feat_y2
        feat_y3 = fmask * gy[2] + (1 - fmask) * feat_y3
        feat_y4 = fmask * gy[3] + (1 - fmask) * feat_y4

        # feat_x1 = tf.Print(feat_x1, [tf.count_nonzero(feat_x1), tf.nn.top_k(feat_x1, 100, sorted=True)[0]], message=' feat x1 : ', summarize=100)

        # feat_ymax = tf.Print(feat_ymax, [tf.shape(feat_ymax), tf.count_nonzero(feat_ymax), feat_ymax,tf.nn.top_k(feat_ymax, 100, sorted=True)[0]], message= ' feat_ymax :', summarize=100)
        # feat_ymin = tf.Print(feat_ymin, [tf.shape(feat_ymin), tf.count_nonzero(feat_ymin), feat_ymin, tf.nn.top_k(feat_ymax, 100, sorted=True)[0]], message= ' feat_ymin :', summarize=100)
        # feat_xmax = tf.Print(feat_xmax, [tf.shape(feat_xmax), tf.count_nonzero(feat_xmax), feat_xmax, tf.nn.top_k(feat_xmax, 100, sorted=True)[0]], message= ' feat_xmax : ', summarize=100)
        # feat_xmin = tf.Print(feat_xmin, [tf.shape(feat_xmin), tf.count_nonzero(feat_xmin), feat_xmin,tf.nn.top_k(feat_xmin, 100, sorted=True)[0]], message= ' feat_xmin: ' , summarize=100)


        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        #mask = tf.logical_and(interscts > ignore_threshold,
        #                     label == no_annotation_label)
        # Replace scores by -1.
        #feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_labels,  feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax,
                feat_x1, feat_x2, feat_x3, feat_x4,
                feat_y1, feat_y2, feat_y3, feat_y4]
    # Main loop definition.

    i = 0
    [i,feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax,
     feat_x1, feat_x2, feat_x3, feat_x4,
     feat_y1, feat_y2, feat_y3, feat_y4] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax, feat_x1, feat_x2, feat_x3, feat_x4, feat_y1, feat_y2, feat_y3, feat_y4])
    # Transform to center / size.
    '''
    这里的逻辑是用gt的外接水平矩形框与anchor/default box做匹配，得到iou的mask之后更新anchor对应的gt
    然后求取anchor对应gt的偏移
    '''
    #
    # feat_ymax = tf.Print(feat_ymax, [tf.shape(feat_ymax), tf.count_nonzero(feat_ymax), feat_ymax], message= ' feat_ymax :', summarize=100)
    # feat_ymin = tf.Print(feat_ymin, [tf.shape(feat_ymin), tf.count_nonzero(feat_ymin), feat_ymin], message= ' feat_ymin :', summarize=100)
    # feat_xmax = tf.Print(feat_xmax, [tf.shape(feat_xmax), tf.count_nonzero(feat_xmax), feat_xmax], message= ' feat_xmax : ', summarize=100)
    # feat_xmin = tf.Print(feat_xmin, [tf.shape(feat_xmin), tf.count_nonzero(feat_xmin), feat_xmin], message= ' feat_xmin: ' , summarize=100)


    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.

    # feat_ymin = tf.Print(feat_ymin, [tf.shape(feat_ymin), feat_ymin], message= ' feat_ymin : ', summarize=20)
    # feat_xmin = tf.Print(feat_xmin, [tf.shape(feat_xmin), feat_xmin], message= ' feat_xmin : ', summarize=20)

    #



    # feat_cy = tf.Print(feat_cy, [tf.shape(feat_cy), feat_cy],message=' feat_cy : ', summarize=20)
    # feat_cx = tf.Print(feat_cx, [tf.shape(feat_cx), feat_cx],message=' feat_cy : ', summarize=20)
    # feat_h = tf.Print(feat_h, [tf.shape(feat_h), feat_h], message=' feat_h : ', summarize=20)
    # feat_w = tf.Print(feat_w, [tf.shape(feat_w), feat_w], message=' feat_w : ', summarize=20)
    #
    # yref = tf.Print(yref, [tf.shape(yref), yref], message=' yref : ',summarize=20)
    # xref = tf.Print(xref, [tf.shape(xref), xref], message=' xref : ',summarize=20)
    # href = tf.Print(href, [tf.shape(href), href], message=' href : ',
    #                 summarize=20)
    # wref = tf.Print(wref, [tf.shape(wref), wref], message=' wref : ', summarize=20)

    feat_ymin = (feat_cy - yref) / href / prior_scaling[1]
    feat_xmin = (feat_cx - xref) / wref / prior_scaling[0]


    feat_ymax = tf.log(feat_h / href) / prior_scaling[3]
    feat_xmax = tf.log(feat_w / wref) / prior_scaling[2]


    feat_x1 = (feat_x1 - xmin) / wref / prior_scaling[0]
    feat_x2 = (feat_x2 - xmax) / wref / prior_scaling[0]
    feat_x3 = (feat_x3 - xmax) / wref / prior_scaling[0]
    feat_x4 = (feat_x4 - xmin) / wref / prior_scaling[0]

    feat_y1 = (feat_y1 - ymin) / href / prior_scaling[1]
    feat_y2 = (feat_y2 - ymin) / href / prior_scaling[1]
    feat_y3 = (feat_y3 - ymax) / href / prior_scaling[1]
    feat_y4 = (feat_y4 - ymax) / href / prior_scaling[1]

    # Use SSD ordering: x / y / w / h instead of ours.
    # add xy1, 2,3,4

    # feat_ymin = tf.Print(feat_ymin, [tf.shape(feat_ymin), feat_ymin], message= ' feat_ymin : ', summarize=20)
    # feat_xmin = tf.Print(feat_xmin, [tf.shape(feat_xmin), feat_xmin], message= ' feat_xmin : ', summarize=20)

    feat_localizations = tf.stack([feat_xmin, feat_ymin, feat_xmax, feat_ymax ,feat_x1, feat_y1, feat_x2, feat_y2, feat_x3, feat_y3, feat_x4, feat_y4], axis=-1)
    # feat_localizations = tf.Print(feat_localizations, [tf.shape(feat_localizations), feat_localizations], message=' feat_localizations: ', summarize=20)
    return feat_labels, feat_localizations, feat_scores



def tf_text_bboxes_encode(glabels, bboxes,
                         anchors,gxs, gys,
                         matching_threshold=0.1,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='text_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      gxs,gys:shape = (N,4) with x,y coordinates
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """

    with tf.name_scope('text_bboxes_encode'):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_label, t_loc, t_scores = \
                    tf_text_bboxes_encode_layer(glabels, bboxes,gxs, gys, anchors_layer,
                                                matching_threshold,
                                               prior_scaling, dtype)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
                target_labels.append(t_label)
        return target_localizations, target_scores, target_labels



