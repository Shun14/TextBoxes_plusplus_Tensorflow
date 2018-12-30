# -*- coding:utf-8 -*-
# from __future__ import unicode_literals
from __future__ import division

import os
import logging
import time
import xml.dom.minidom

import numpy as np
import tensorflow as tf
import cv2
import tensorflow.contrib.slim as slim
import codecs
from nets import np_methods, txtbox_384
from processing import ssd_vgg_preprocessing
from eval import EVAL_MODEL
from argparse import ArgumentParser


class TextboxesDetection(object):
    def __init__(self,
                 model_dir,
                 in_dir,
                 out_dir,
                 nms_th_for_all_scale=0.5,
                 score_th=0.2,
                 scales=([384, 384], [768, 384], [768, 768]),
                 min_side_scale=384,
                 save_res_path='eval_res.txt'
                 ):
        if os.path.exists(in_dir):
            self.in_dir = in_dir
        else:
            raise ValueError('{} does not existed!!!'.format(in_dir))

        self.out_dir = out_dir
        self.suffixes = ['.png', '.PNG', '.jpg', '.jpeg']

        self.img_path, self.img_num = self.get_img_path()

        self.nms_th_for_all_scale = nms_th_for_all_scale
        self.nms_threshold = 0.45
        self.score_th = score_th
        print('self.score_th', self.score_th)
        self.make_out_dir()
        self.text_scales = scales
        self.data_format = 'NHWC'
        self.select_threshold = 0.01
        self.min_side_scale = min_side_scale
        self.max_side_scale = self.min_side_scale * 2  # 384 * 2
        self.save_xml_flag = True
        self.save_txt_flag = True
        self.dynamic_scale_flag = False
        self.allow_padding = False
        self.allow_post_processing = False
        self.allow_eval_flag = False
        self.resize_flag = False
        self.save_eval_resut_path = save_res_path
        self.model_path = None

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True

        self.graph = tf.Graph()
        self.session_text = tf.Session(graph=self.graph, config=self.config)

        with self.session_text.as_default():
            with self.graph.as_default():
                self.img_text = tf.placeholder(
                    tf.float32, shape=(None, None, 3))
                print(len(self.text_scales))
                self.scale_text = tf.placeholder(tf.int32, shape=(2))

                img_pre_text, label_pre_text, bboxes_pre_text, self.bboxes_img_text, xs_text, ys_text = ssd_vgg_preprocessing.preprocess_for_eval(
                    self.img_text,
                    None,
                    None,
                    None,
                    None,
                    self.scale_text,
                    self.data_format,
                    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
                image_text_4d = tf.expand_dims(img_pre_text, 0)
                image_text_4d = tf.cast(image_text_4d, tf.float32)
                self.image_text_4d = image_text_4d
                self.net_text = txtbox_384.TextboxNet()
                with slim.arg_scope(
                        self.net_text.arg_scope(data_format=self.data_format)):
                    self.predictions_text, self.localisations_text, self.logits_text, self.endpoints_text, self.l_shape = self.net_text.net(
                        self.image_text_4d,
                        is_training=False,
                        reuse=tf.AUTO_REUSE,
                        update_feat_shapes=True)
                saver_text = tf.train.Saver()
                if os.path.isdir(model_dir):
                    ckpt_path = tf.train.latest_checkpoint(model_dir)
                    self.model_path = os.path.join(model_dir, ckpt_path)
                else:
                    ckpt_path = model_dir
                    self.model_path = ckpt_path
                print(model_dir)
                saver_text.restore(self.session_text, ckpt_path)

        logging.info("Textbox++ model initialized.")

    def make_out_dir(self):
        out_dir = self.out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        save_rbox_dir = os.path.join(
            out_dir, 'public_polygon_multi', '{}_{}'.format(
                self.score_th, self.nms_th_for_all_scale))
        save_nms_dir = os.path.join(
            out_dir, 'multi_scale_nms_public', '{}_{}'.format(
                self.score_th,
                self.nms_th_for_all_scale))  # path for save NMS results
        save_visu_dir = os.path.join(
            out_dir, 'multi_scale_visu_public', '{}_{}'.format(
                self.score_th, self.nms_th_for_all_scale
            ))  # path for save visualization images

        if not os.path.exists(save_nms_dir):
            os.makedirs(save_nms_dir)

        if not os.path.exists(save_rbox_dir):
            os.makedirs(save_rbox_dir)

        if not os.path.exists(save_visu_dir):
            os.makedirs(save_visu_dir)

        self.save_rbox_dir = save_rbox_dir
        self.save_nms_dir = save_nms_dir
        self.save_visu_dir = save_visu_dir

    def get_img_path(self):
        file_num = 0
        file_names = []
        for root, dirs, files in os.walk(self.in_dir):
            for file in files:
                file_name = os.path.join(root, file)
                if os.path.splitext(file_name)[1] in self.suffixes:
                    file_num += 1
                    file_names.append(file_name)
        return file_names, file_num

    def get_all_img_info(self):
        img_list, img_num = self.img_path, self.img_num
        print('all num:', img_num)
        return img_list, img_num

    def judge_pic_scale(self, img_in):
        h, w, _ = img_in.shape
        resize_w = w
        resize_h = h

        min_side_scale = self.min_side_scale
        ratio = float(resize_h) / resize_w
        # ratio = int(ratio) if ratio > 1. else int(1.0 / ratio)
        if ratio < 1.:
            ratio = int(ratio) if ratio > 1. else int(1.0 / ratio)
            scales = [[min_side_scale, min_side_scale * ratio]]
        elif ratio > 1.:
            ratio = int(ratio) if ratio > 1. else int(1.0 / ratio)
            scales = [[min_side_scale * ratio, min_side_scale]]
        elif ratio == 1 and resize_h < min_side_scale and resize_w < min_side_scale:
            scales = [[min_side_scale, min_side_scale]]

        return scales

    def padding_for_scale(self, scale, img_ori):
        h_ori, w_ori, _ = img_ori.shape
        h_scale = scale[0]
        w_scale = scale[1]
        padding_flag = True
        color = [255, 255, 255]
        top = 0
        bottom = 0
        left = 0
        right = 0

        if h_ori < h_scale and w_ori < w_scale:
            delta_h = h_scale - h_ori
            delta_w = w_scale - w_ori
            top, bottom = int(
                float(delta_h) / 2), int(delta_h - float(delta_h) / 2)
            left, right = int(
                float(delta_w) / 2), int(delta_w - float(delta_w) / 2)
            h_scale = h_ori + top + bottom
            w_scale = w_ori + left + right

            new_im = cv2.copyMakeBorder(
                img_ori,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=color)

        elif h_ori < h_scale:
            delta_h = h_scale - h_ori
            top, bottom = int(
                float(delta_h) / 2), int(delta_h - float(delta_h) / 2)
            h_scale = h_ori + top + bottom
            w_scale = w_ori
            new_im = cv2.copyMakeBorder(
                img_ori,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=color)

        elif w_ori < w_scale:
            delta_w = w_scale - w_ori
            left, right = int(
                float(delta_w) / 2), int(delta_w - float(delta_w) / 2)
            new_im = cv2.copyMakeBorder(
                img_ori,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=color)
            w_scale = w_ori + left + right
            h_scale = h_ori

        else:
            new_im = img_ori
            padding_flag = False

        return padding_flag, new_im, h_scale, w_scale, left, top

    def resize_and_padding_for_scale(self, scale,
                                     ori_img):  # scale > ori_img.shape
        h_ori, w_ori, _ = ori_img.shape
        ratio = float(h_ori) / w_ori
        resize_ratio = 1.
        if h_ori > self.max_side_scale or w_ori > self.max_side_scale:
            self.resize_flag = True

        if self.resize_flag is True:
            if ratio > 1.:
                resize_ratio = scale[0] / float(h_ori)
            else:
                resize_ratio = scale[1] / float(w_ori)
        else:
            resize_ratio = 1.
        h_resize = int(resize_ratio * h_ori)
        w_resize = int(resize_ratio * w_ori)
        img_resize = cv2.resize(ori_img, (w_resize, h_resize))
        padding_flag, new_im, h_scale, w_scale, left, top = self.padding_for_scale(
            scale, img_resize)

        return padding_flag, new_im, h_scale, w_scale, left, top, h_resize, w_resize

    def start_inference(self):
        software_start_time = time.time()
        img_list, img_num = self.img_path, self.img_num
        i = 0
        print('*******start inference**********')
        all_time = 0.0
        for img_file_path in img_list:
            img_in = cv2.imread(img_file_path)

            img_name = img_file_path.split('/')[-1][:-4]

            start_time = time.time()
            print(img_file_path)
            if self.dynamic_scale_flag:
                self.text_scales = self.judge_pic_scale(img_in)
                print(self.judge_pic_scale(img_in))

            outputs = self.detect_text(img_in, img_name)

            nms_outputs = self.nms_single(outputs, img_name)
            use_time = time.time() - start_time
            print('use_time: {}'.format(use_time))
            if self.save_xml_flag is True:
                self.save_xml_file(nms_outputs, img_name, img_in)
            self.save_vis_pic(nms_outputs, img_name, img_in)
            all_time += use_time
            i += 1
            print('{}:{}'.format(i, img_num))
        software_end_time = time.time() - software_start_time
        print('**********end******************')
        print('all img time use:', all_time, 'one pic time use:',
              all_time / img_num)
        print('code run:', software_end_time)

        if self.allow_eval_flag is True and self.save_eval_resut_path is not None:
            eval_model = EVAL_MODEL(self.in_dir, self.save_nms_dir, '1',
                                    self.save_eval_resut_path)
            if len(eval_model.get_xml_path()) != 0:
                eval_model.start_eval()
                with open(self.save_eval_resut_path, 'a+') as f:
                    f.write('nms_th:{} score_th:{} model-name:{}\n'.format(self.nms_th_for_all_scale, self.score_th,self.model_path))

    def save_xml_file(self, nms_outputs, img_name, img):
        dt_lines = [l.strip() for l in nms_outputs]
        dt_lines = [list_from_str(dt)
                    for dt in dt_lines]  # score xmin ymin xmax ymax

        annotation_xml = xml.dom.minidom.Document()
        root = annotation_xml.createElement('annotation')
        annotation_xml.appendChild(root)

        nodeFolder = annotation_xml.createElement('folder')
        root.appendChild(nodeFolder)

        nodeFilename = annotation_xml.createElement('filename')
        nodeFilename.appendChild(
            annotation_xml.createTextNode('{}.png'.format(img_name)))
        root.appendChild(nodeFilename)

        nodeSize = annotation_xml.createElement('size')
        nodeWidth = annotation_xml.createElement('width')
        nodeHeight = annotation_xml.createElement('height')
        nodeDepth = annotation_xml.createElement('depth')

        nodeWidth.appendChild(annotation_xml.createTextNode(str(img.shape[1])))
        nodeHeight.appendChild(
            annotation_xml.createTextNode(str(img.shape[0])))
        nodeDepth.appendChild(annotation_xml.createTextNode(str(img.shape[2])))

        nodeSize.appendChild(nodeWidth)
        nodeSize.appendChild(nodeHeight)
        nodeSize.appendChild(nodeDepth)
        root.appendChild(nodeSize)

        for dt in dt_lines:
            xmin_text = str(int(dt[1]))
            ymin_text = str(int(dt[2]))
            xmax_text = str(int(dt[3]))
            ymax_text = str(int(dt[4]))

            nodeObject = annotation_xml.createElement('object')
            nodeDifficult = annotation_xml.createElement('difficult')
            nodeDifficult.appendChild(annotation_xml.createTextNode('0'))
            nodeName = annotation_xml.createElement('name')
            nodeName.appendChild(annotation_xml.createTextNode('1'))

            nodeBndbox = annotation_xml.createElement('bndbox')
            nodexmin = annotation_xml.createElement('xmin')
            nodexmin.appendChild(annotation_xml.createTextNode(xmin_text))
            nodeymin = annotation_xml.createElement('ymin')
            nodeymin.appendChild(annotation_xml.createTextNode(ymin_text))
            nodexmax = annotation_xml.createElement('xmax')
            nodexmax.appendChild(annotation_xml.createTextNode(xmax_text))
            nodeymax = annotation_xml.createElement('ymax')
            nodeymax.appendChild(annotation_xml.createTextNode(ymax_text))

            nodeBndbox.appendChild(nodexmin)
            nodeBndbox.appendChild(nodeymin)
            nodeBndbox.appendChild(nodexmax)
            nodeBndbox.appendChild(nodeymax)
            nodeObject.appendChild(nodeDifficult)
            nodeObject.appendChild(nodeName)
            nodeObject.appendChild(nodeBndbox)
            root.appendChild(nodeObject)

        xml_path = os.path.join(self.save_nms_dir, img_name + '.xml')
        fp = open(xml_path, 'w')
        annotation_xml.writexml(
            fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

    def save_vis_pic(self, nms_outputs, img_name, img):
        dt_lines = [l.strip() for l in nms_outputs]
        dt_lines = [list_from_str(dt) for dt in dt_lines]

        for dt in dt_lines:
            img = self.draw_polygon(img, dt[1], dt[2], dt[3], dt[4])
        # img_save_path = os.path.join(self.save_vis_pic, '{}.png'.format(img_name))
        img_save_path = '{}/{}.png'.format(self.save_visu_dir, img_name)
        print(img_save_path)
        cv2.imwrite(img_save_path, img)

    def nms_single(self, all_scale_res, img_name):
        dt_lines = [l.strip() for l in all_scale_res]
        dt_lines = [list_from_str(dt) for dt in dt_lines]
        dt_lines = sorted(dt_lines, key=lambda x: -float(x[0]))
        nms_flag, dt_lines_new = nms_eff(dt_lines, self.score_th,
                                         self.nms_th_for_all_scale)
        boxes = []
        for k, dt, in enumerate(dt_lines_new):
            if nms_flag[k]:
                if dt[0] > self.score_th:
                    if dt not in boxes:
                        boxes.append(dt)

        mean_value = 0.0
        line_count = 0
        final_bboxes = []
        if self.allow_post_processing:
            # 这里是为了将两个水平线上重叠的框连接起来，解决长文本漏掉的情况,该处的后处理和eval一样
            del_index = []
            for i, box in enumerate(boxes):
                if i in del_index:
                    continue
                if len(boxes[i + 1:]) == 0:
                    if box not in final_bboxes and i not in del_index:
                        final_bboxes.append(box)
                    break
                for j, box_2rd in enumerate(boxes[i + 1:]):
                    if j in del_index:
                        continue
                    score_second = float(box_2rd[0])
                    ymin_second = int(box_2rd[2])
                    ymax_second = int(box_2rd[4])
                    xmin_second = int(box_2rd[1])
                    xmax_second = int(box_2rd[3])
                    ymin_first = int(box[2])
                    ymax_first = int(box[4])
                    xmin_first = int(box[1])
                    xmax_first = int(box[3])
                    score_first = float(box[0])
                    if abs(ymin_second - ymin_first) <= 5 and abs(
                            ymax_first - ymax_second) <= 5 and mat_inter(
                                box, box_2rd):
                        xmin_final = min(xmin_first, xmin_second)
                        ymin_final = min(ymin_first, ymin_second)
                        xmax_final = max(xmax_first, xmax_second)
                        ymax_final = max(ymax_first, ymax_second)
                        temp_box = [(score_first + score_second) / float(2),
                                    xmin_final, ymin_final, xmax_final,
                                    ymax_final]
                        del_index.append(i)
                        del_index.append(j + i + 1)
                        box = temp_box
                final_bboxes.append(box)
            boxes = final_bboxes

        nms_outputs = []
        for box in boxes:
            box = [b for b in box]
            line_count += 1
            mean_value += float(box[0])
            nms_outputs.append('text {} {} {} {} {}\n'.format(
                str(float(box[0])), str(int(box[1])), str(int(box[2])),
                str(int(box[3])), str(int(box[4]))))

        if self.save_txt_flag is True:
            with codecs.open(
                    '{}/{}.txt'.format(self.save_nms_dir, img_name),
                    'w',
                    encoding='utf-8') as f:
                f.writelines(nms_outputs)
        return nms_outputs

    def detect_text(self, img_in, img_name):
        all_boxes = []
        all_scores = []
        all_classes = []
        for scale in self.text_scales:
            print('scale:', scale)
            if self.allow_padding:
                #对图片resize and padding
                padding_flag, new_im, h_scale, w_scale, left, top, h_resize, w_resize = self.resize_and_padding_for_scale(
                    scale, img_in)
                if padding_flag is True:
                    #img_in = new_im
                    h_ori, w_ori = h_resize, w_resize  #
                    # h_ori, w_ori , _= img_in.shape
                    print(h_scale, w_scale, h_ori, w_ori, left, top)
                    rclasses, rscores, rbboxes = self.detect_single_scale(
                        new_im, scale, name=img_name)
                    rbboxes[:, 4] = (
                        rbboxes[:, 4] * float(w_scale) - left) / w_ori
                    rbboxes[:, 5] = (
                        rbboxes[:, 5] * float(w_scale) - left) / w_ori
                    rbboxes[:, 6] = (
                        rbboxes[:, 6] * float(w_scale) - left) / w_ori
                    rbboxes[:, 7] = (
                        rbboxes[:, 7] * float(w_scale) - left) / w_ori

                    rbboxes[:,
                            8] = (rbboxes[:, 8] * float(h_scale) - top) / h_ori
                    rbboxes[:,
                            9] = (rbboxes[:, 9] * float(h_scale) - top) / h_ori
                    rbboxes[:, 10] = (
                        rbboxes[:, 10] * float(h_scale) - top) / h_ori
                    rbboxes[:, 11] = (
                        rbboxes[:, 11] * float(h_scale) - top) / h_ori

                else:
                    rclasses, rscores, rbboxes = self.detect_single_scale(
                        img_in, scale, name=img_name)
            else:
                rclasses, rscores, rbboxes = self.detect_single_scale(
                    img_in, scale, name=img_name)

            all_boxes.extend(rbboxes)
            all_scores.extend(rscores)
            all_classes.extend(rclasses)

        all_classes = np.array(all_classes)
        all_scores = np.array(all_scores)
        all_boxes = np.array(all_boxes)
        all_classes, all_scores, all_boxes = np_methods.bboxes_sort(
            all_classes, all_scores, all_boxes, top_k=400)
        if len(all_boxes) == 0:
            return []
        all_classes, all_scores, all_boxes = np_methods.bboxes_nms(
            all_classes,
            all_scores,
            all_boxes,
            nms_threshold=self.nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        # all_boxes = np_methods.bboxes_resize(img_in, all_boxes)

        outputs = []
        img_height, img_width, _ = img_in.shape
        image = img_in
        det_xmin = all_boxes[:, 0]
        det_ymin = all_boxes[:, 1]
        det_xmax = all_boxes[:, 2]
        det_ymax = all_boxes[:, 3]

        det_x1 = all_boxes[:, 4]
        det_x2 = all_boxes[:, 5]
        det_x3 = all_boxes[:, 6]
        det_x4 = all_boxes[:, 7]
        det_y1 = all_boxes[:, 8]
        det_y2 = all_boxes[:, 9]
        det_y3 = all_boxes[:, 10]
        det_y4 = all_boxes[:, 11]
        # print(all_boxes)
        for i in range(all_scores.shape[0]):
            x1 = int(round(det_x1[i] * image.shape[1]))
            y1 = int(round(det_y1[i] * image.shape[0]))
            x2 = int(round(det_x2[i] * image.shape[1]))
            y2 = int(round(det_y2[i] * image.shape[0]))
            x3 = int(round(det_x3[i] * image.shape[1]))
            y3 = int(round(det_y3[i] * image.shape[0]))
            x4 = int(round(det_x4[i] * image.shape[1]))
            y4 = int(round(det_y4[i] * image.shape[0]))

            x1 = max(0, min(x1, image.shape[1] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            x3 = max(0, min(x3, image.shape[1] - 1))
            x4 = max(0, min(x4, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))
            y3 = max(0, min(y3, image.shape[0] - 1))
            y4 = max(0, min(y4, image.shape[0] - 1))
            # #cx
            # xmin = int(round(det_xmin[i] * image.shape[1]))
            # xmin = max(0, min(xmin, image.shape[1] - 1))
            # #cy
            # ymin = int(round(det_ymin[i] * image.shape[0]))
            # ymin = max(0, min(ymin, image.shape[0] - 1))
            # #cw
            # xmax = int(round(det_xmax[i] * image.shape[1]))
            # xmax = max(0, min(xmax, image.shape[1] - 1))
            # #ch
            # ymax = int(round(det_ymin[i] * image.shape[0]))
            # ymax = max(0, min(ymax, image.shape[0] - 1))

            # xmin_final = xmin - xmax / 2
            # ymin_final = ymin - ymax / 2
            # xmax_final = xmin + xmax / 2
            # ymax_final = ymin + ymax / 2

            # xmin = xmin_final
            # ymin = ymin_final
            # xmax = xmax_final
            # ymax = ymax_final
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            # error box
            if ymin == ymax:
                continue
            score = all_scores[i]
            outputs.append('text {} {} {} {} {}\n'.format(
                score, xmin, ymin, xmax, ymax))

        with codecs.open(
                '{}/{}.txt'.format(self.save_rbox_dir, img_name),
                'w',
                encoding='utf-8') as f:
            f.writelines(outputs)
        return outputs

    def textboxes_feat_shapes_from_net(self, l_shape, default_shapes=None):
        feat_shapes = []
        for l in l_shape:
            shape = tuple(l[1:3])

            if None in shape:
                return default_shapes
            else:
                feat_shapes.append(shape)
        return feat_shapes

    def detect_single_scale(self, img_in, scale, name):
        rimage_text, rpredictions_text, rlocalisations_text, l_shape_text = self.session_text.run(
            [
                self.image_text_4d, self.predictions_text,
                self.localisations_text, self.l_shape
            ],
            feed_dict={
                self.img_text: img_in,
                self.scale_text: scale
            })
        # cv2.imwrite('{}_input_{}.png'.format(os.path.join(self.save_visu_dir, name), scale), rimage_text[0])
        shapes = self.textboxes_feat_shapes_from_net(
            l_shape_text, self.net_text.params.feat_shapes)
        self.net_text.params = self.net_text.params._replace(
            feat_shapes=shapes)
        txt_anchors = self.net_text.anchors(scale)
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions_text,
            rlocalisations_text,
            txt_anchors,
            select_threshold=self.select_threshold,
            img_shape=scale,
            num_classes=2,
            decode=True)

        return rclasses, rscores, rbboxes

    def draw_polygon(self, img, xmin, ymin, xmax, ymax):
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        cv2.line(img, (xmin, ymin), (xmax, ymin), (255, 0, 0), 2)
        cv2.line(img, (xmax, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.line(img, (xmax, ymax), (xmin, ymax), (255, 0, 0), 2)
        cv2.line(img, (xmin, ymax), (xmin, ymin), (255, 0, 0), 2)
        return img

    def draw_point(self, img, x1, y1):
        cv2.line(img, (x1, y1), (x1 + 3, y1), (0, 0, 255), 5)
        return img


def list_from_str(st, dtype='float32'):
    line = st.split(' ')[1:6]
    if dtype == 'float32':
        line = [float(a) for a in line]
    else:
        line = [int(a) for a in line]
    return line


def mat_inter(box1, box2):
    _, xmin_1, ymin_1, xmax_1, ymax_1 = box1
    _, xmin_2, ymin_2, xmax_2, ymax_2 = box2
    distance_between_box_x = abs((xmax_1 + xmin_1) / 2 - (xmax_2 + xmin_2) / 2)
    distance_between_box_y = abs((ymax_2 + ymin_2) / 2 - (ymin_1 + ymax_2) / 2)

    distance_box_1_x = abs(xmin_1 - xmax_1)
    distance_box_1_y = abs(ymax_1 - ymin_1)
    distance_box_2_x = abs(xmax_2 - xmin_2)
    distance_box_2_y = abs(ymax_2 - ymin_2)

    if distance_between_box_x < (distance_box_1_x + distance_box_2_x
                                 ) / 2 and distance_between_box_y < (
                                     distance_box_2_y + distance_box_1_y) / 2:
        return True
    else:
        return False


def nms_eff(boxes_list, score, overlap):
    nms_flag = [False] * len(boxes_list)
    boxes = np.array(boxes_list)
    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    scores = boxes[:, 0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and scores[order[0]] > score:
        i = order[0]
        nms_flag[i] = True
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # ovr = inter / (areas[i] + areas[order[1:]] - inter)
        areas_cand = areas[order[1:]]
        areas_tmp = np.zeros_like(areas_cand)
        areas_tmp[:] = areas[i]
        areas_min = np.min(np.vstack((areas_cand, areas_tmp)), axis=0)
        ovr = inter / areas_min

        inds = np.where(ovr <= overlap)[0]
        order = order[inds + 1]
    return nms_flag, boxes_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='icdar15 test model')
    parser.add_argument(
        '--in_dir',
        '-i',
        default=
        '/home/zsz/datasets/text_det/icdar15/test_images/',
        type=str)
    parser.add_argument(
        '--out_dir', '-o', default='icdar15_scale_test/3000_with', type=str)
    parser.add_argument(
        '--model_dir',
        '-m',
        default=
        '/home/zsz/test/dssd_tfmodel/only_10th_data_with_the_new_pic',
        type=str)
    parser.add_argument('--cuda_device', '-c', default='3', type=str)
    parser.add_argument('--nms_th', '-n', default=0.5, type=float)
    parser.add_argument('--score_th','-s', default=0.1, type=float)
    parser.add_argument('--save_res_path', '-r', default='eval_res.txt', type=str)
    #2:read gt from txt format:(text score xmin ymin xmax ymax)
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    model_dir = args.model_dir
    nms_th = args.nms_th
    score_th = args.score_th
    save_res_path = args.save_res_path
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)  # using GPU 3

    instance = TextboxesDetection(
        model_dir,
        in_dir,
        out_dir,
        nms_th,
        score_th,
        scales=[(384, 384)],
        save_res_path=save_res_path
        )
    instance.start_inference()
