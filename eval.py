#! -*- encoding: utf-8 -*-
import numpy as np
import cv2
import os
from argparse import ArgumentParser

import xml.etree.ElementTree as ET
import shutil
import sys
import matplotlib.pyplot as plt
info = sys.version_info
if int(info[0]) == 2:
    reload(sys)
    sys.setdefaultencoding('utf-8')  # 设置 'utf-8'


def mat_inter(box1, box2):
    xmin_1, ymin_1, xmax_1, ymax_1 = box1
    xmin_2, ymin_2, xmax_2, ymax_2 = box2
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


class EVAL_MODEL(object):
    def __init__(self,
                 eval_data_dir,
                 pre_data_dir,
                 data_type,
                 save_result_path,
                 iou_th=0.5,
                 save_err_path='err_pic'):
        # print(eval_data_dir , pre_data_dir ,data_type, save_result_path)
        if eval_data_dir is None or pre_data_dir is None or data_type is None or save_result_path is None:
            raise ValueError(
                'please input eval_data_dir or pre_data_dir or data_type or save_result_path'
            )
        self.eval_data_dir = eval_data_dir
        self.pre_data_dir = pre_data_dir
        self.data_type = data_type
        self.save_result_path = save_result_path

        self.allow_post_processing = False
        self.draw_err_pic_flag = True
        self.xml_path_list = []
        self.pre_img_name_list = []
        self.eval_data_dict = {}
        self.pre_data_dict = {}

        self.pre_data_num = 0
        self.gt_data_num = 0
        self.hit_precision = 0
        self.hit_recall = 0
        self.err_gt_dict = {
            'ratio': [],
            'h_scale': [],
            'w_scale': [],
            'height': [],
            'width': []
        }  #save box ratio and size
        self.iou_thresh = float(iou_th)
        self.save_err_path = os.path.join(save_err_path, str(self.iou_thresh))
        if not os.path.exists(self.save_err_path):
            os.makedirs(self.save_err_path)

    def list_from_str(self, st, dtype='float32'):
        line = st.split(' ')[2:6]
        if dtype == 'float32':
            line = [float(a) for a in line]
        else:
            line = [int(a) for a in line]
        return line

    def get_xml_path(self):
        for i in os.listdir(self.eval_data_dir):
            if i.split('.')[-1] == 'xml':
                self.xml_path_list.append(os.path.join(self.eval_data_dir, i))
        return self.xml_path_list

    def read_gts(self):
        if self.data_type == '1':
            if self.eval_data_dir is None or self.eval_data_dir is '':
                raise ValueError('---eval data dir not exists!!!!-----')
            for i in self.xml_path_list:
                img_name = os.path.splitext(os.path.basename(i))[0]
                # img_name = os.path.basename(i)
                xml_info = ET.parse(i)
                root_node = xml_info.getroot()
                bbox_list = []
                for obj_node in root_node.findall('object'):
                    name_node = obj_node.find('name')
                    # print(name_node)
                    name = name_node.text
                    if name == '&*@HUST_special' or name == '&*@HUST_shelter':
                        continue
                    difficult = int(obj_node.find('difficult').text)
                    if difficult == 1:
                        continue

                    bndbox_node = obj_node.find('bndbox')
                    xmin_filter = int(bndbox_node.find('xmin').text)
                    ymin_filter = int(bndbox_node.find('ymin').text)
                    xmax_filter = int(bndbox_node.find('xmax').text)
                    ymax_filter = int(bndbox_node.find('ymax').text)

                    bbox_list.append(
                        [xmin_filter, ymin_filter, xmax_filter, ymax_filter])

                self.eval_data_dict[img_name] = bbox_list
        elif self.data_type == '2':
            pass
        else:
            raise ValueError('  data type error !!! ')

    #list format: xmin ,ymin, xmax, ymax
    def bbox_iou_eval(self, list1, list2):
        xx1 = np.maximum(list1[0], list2[0])
        yy1 = np.maximum(list1[1], list2[1])
        xx2 = np.minimum(list1[2], list2[2])
        yy2 = np.minimum(list1[3], list2[3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        area1 = (list1[2] - list1[0] + 1) * (list1[3] - list1[1] + 1)
        area2 = (list2[2] - list2[0] + 1) * (list2[3] - list2[1] + 1)
        iou = float(inter) / (area1 + area2 - inter)
        return iou

    def read_pres(self):
        for i in os.listdir(self.pre_data_dir):
            if i.split('.')[-1] == 'txt':
                img_name = os.path.splitext(os.path.basename(i))[0]
                # img_name = os.path.basename(i)
                self.pre_img_name_list.append(img_name)
                nms_outputs = open(os.path.join(self.pre_data_dir, i),
                                   'r').readlines()
                dt_lines = [l.strip() for l in nms_outputs]
                dt_lines = [
                    self.list_from_str(dt, dtype='int32') for dt in dt_lines
                ]  # score xmin ymin xmax ymax
                boxes = dt_lines
                bbox_without_same = []
                for box in boxes:
                    if box[1] != box[3]:
                        bbox_without_same.append(box)
                boxes = bbox_without_same
                final_bboxes = []
                if self.allow_post_processing is True:
                    #box format : score xmin  ymin xmax ymax
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
                            ymin_second = int(box_2rd[1])
                            ymax_second = int(box_2rd[3])
                            xmin_second = int(box_2rd[0])
                            xmax_second = int(box_2rd[2])
                            ymin_first = int(box[1])
                            ymax_first = int(box[3])
                            xmin_first = int(box[0])
                            xmax_first = int(box[2])
                            if abs(ymin_second - ymin_first) <= 5 and abs(
                                    ymax_first -
                                    ymax_second) <= 5 and mat_inter(
                                        box, box_2rd):

                                xmin_final = min(xmin_first, xmin_second)
                                ymin_final = min(ymin_first, ymin_second)
                                xmax_final = max(xmax_first, xmax_second)
                                ymax_final = max(ymax_first, ymax_second)
                                temp_box = [
                                    xmin_final, ymin_final, xmax_final,
                                    ymax_final
                                ]
                                del_index.append(i)
                                del_index.append(j + i + 1)
                                box = temp_box
                        final_bboxes.append(box)
                    dt_lines = final_bboxes

                self.pre_data_dict[img_name] = dt_lines

    def contrast_pre_gt(self):
        for img_name in self.pre_img_name_list:
            pre_gts = self.pre_data_dict[img_name]
            eval_gts = self.eval_data_dict[img_name]
            error_pre = []
            error_eval = []

            for i, eval_gt in enumerate(eval_gts):
                flag_strick = 0
                err_flag = False
                for j, pre_gt in enumerate(pre_gts):
                    iou = self.bbox_iou_eval(eval_gt, pre_gt)
                    # print(iou)
                    if iou >= self.iou_thresh:
                        flag_strick += 1
                        err_flag = False
                        break
                    else:
                        err_flag = True
                if flag_strick >= 1:
                    self.hit_precision += 1
                    self.hit_recall += 1
                self.gt_data_num += 1
                if err_flag is True:
                    error_eval.append(eval_gt)

            self.pre_data_num += len(pre_gts)

            for i, pre_gt in enumerate(pre_gts):
                err_flag = False
                for j, eval_gt in enumerate(eval_gts):
                    iou = self.bbox_iou_eval(pre_gt, eval_gt)
                    if iou >= self.iou_thresh:
                        err_flag = False
                        break
                    else:
                        err_flag = True
                if err_flag is True:
                    error_pre.append(pre_gt)
            if len(error_eval) != 0 or len(error_pre) != 0:
                self.save_error_eval(img_name, error_eval, error_pre)

    def save_error_eval(self, img_name, error_eval_list, error_pre_list):
        # img_path = os.path.join(self.eval_data_dir, img_name)
        img_path = os.path.join(self.eval_data_dir, img_name + '.jpg')
        img = cv2.imread(img_path)

        if img is None:
            img_path = os.path.join(self.eval_data_dir, img_name + '.PNG')
            img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        with open(
                os.path.join(self.save_err_path, img_name + '_err_gt.txt'),
                'w') as f_target:
            for err_eval in error_eval_list:
                # print(err_eval)
                width = int(err_eval[2]) - int(err_eval[0])
                height = int(err_eval[3]) - int(err_eval[1])
                if width != 0:
                    self.err_gt_dict['ratio'].append(float(height) / width)
                else:
                    self.err_gt_dict['ratio'].append(0.)
                self.err_gt_dict['h_scale'].append(float(height) / img_h)
                self.err_gt_dict['w_scale'].append(float(width) / img_w)
                self.err_gt_dict['height'].append(height)
                self.err_gt_dict['width'].append(width)
                img = self.draw_polygon(
                    img,
                    err_eval[0],
                    err_eval[1],
                    err_eval[2],
                    err_eval[3],
                    is_gt=True)

                f_target.write(','.join([str(i) for i in err_eval]) + '\n')
        with open(
                os.path.join(self.save_err_path, img_name + '_err_pre.txt'),
                'w') as f_target:
            for err_pre in error_pre_list:
                f_target.write(','.join([str(j) for j in err_pre]) + '\n')
                img = self.draw_polygon(
                    img,
                    err_pre[0],
                    err_pre[1],
                    err_pre[2],
                    err_pre[3],
                    is_gt=False)
        cv2.imwrite(os.path.join(self.save_err_path, img_name + '.png'), img)

    def draw_polygon(self, img, xmin, ymin, xmax, ymax, is_gt=False):
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        if is_gt is True:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.line(img, (xmin, ymin), (xmax, ymin), color, 2)
        cv2.line(img, (xmax, ymin), (xmax, ymax), color, 2)
        cv2.line(img, (xmax, ymax), (xmin, ymax), color, 2)
        cv2.line(img, (xmin, ymax), (xmin, ymin), color, 2)
        return img

    def cal_precision_recall(self):
        recall = float(self.hit_recall) / self.gt_data_num
        precision = float(self.hit_precision) / self.pre_data_num
        if recall != 0 and precision != 0:
            f_measure = 2 * recall * precision / (recall + precision)
        else:
            f_measure = 0
        return [precision, recall, f_measure]

    def draw_err_gt(self):
        err_gt_dict = self.err_gt_dict
        for key in err_gt_dict.keys():
            values = err_gt_dict[key]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            numBins = 50
            (counts, bins, patch) = ax.hist(
                values, numBins, color='blue', alpha=0.4, rwidth=0.5)
            #print('*****', key, '******')
            #print(counts)
            #print(bins)
            for i in range(len(counts)):
                plt.text(
                    bins[i],
                    counts[i],
                    str(int(counts[i])),
                    fontdict={
                        'size': 6,
                        'color': 'r'
                    })
                if key in ['h_scale', 'w_scale', 'ratio']:
                    mid = round((float(bins[i]) + float(bins[i + 1])) / 2, 2)
                else:
                    mid = int(bins[i] + bins[i + 1] / 2)
                #if i % 5 == 0:
                plt.text(
                    bins[i],
                    counts[i] + 20,
                    str(mid),
                    fontdict={
                        'size': 10,
                        'color': 'b'
                    })
            #print(patch)
            plt.grid(True)
            plt.title(u'{}'.format(key))
            plt.savefig('{}/{}.png'.format(self.save_err_path, key))
            with open('{}/{}.txt'.format(self.save_err_path, key), 'w') as f:
                for value in values:
                    f.write('{}\n'.format(value))

    def start_eval(self):
        print('----start eval----')
        print('---get xml path---')
        self.get_xml_path()
        print('---reading gts----')
        self.read_gts()
        print('---reading predictions---')
        self.read_pres()
        print('---contrast pre gt-----')
        self.contrast_pre_gt()
        pre, recall, f_measure = self.cal_precision_recall()
        print('pre:{} recall:{} f_measure:{}'.format(pre, recall, f_measure))
        with open(self.save_result_path, 'a+') as f:
            f.write('iou:{} pre:{} recall:{} f_measure:{}\n'.format(
                self.iou_thresh, pre, recall, f_measure))

        if self.draw_err_pic_flag == True:
            self.draw_err_gt()
        print('-----end eval------')


if __name__ == '__main__':

    parser = ArgumentParser(description='icdar15 eval model')
    parser.add_argument(
        '--eval_data_dir',
        '-d',
        default=
        '/home/zsz/datasets/icdar15/test_gts/',
        type=str)
    #xml and img in same dir
    parser.add_argument('--pre_data_dir', '-p', type=str)
    #pre_data_dir is prediction format txt: text score xmin ymin xmax ymax
    parser.add_argument('--eval_file_type', '-f', default='1', type=str)
    parser.add_argument(
        '--save_result_path', '-s', default='result.txt', type=str)
    parser.add_argument('--iou_th', '-o', default=0.5, type=float)
    #1:read gt from xml format:(xmin ymin xmax ymax)
    #2:read gt from txt format:(text score xmin ymin xmax ymax)
    args = parser.parse_args()
    # print(args, args.eval_data_dir)
    eval_data_dir = args.eval_data_dir
    pre_data_dir = args.pre_data_dir
    eval_file_type = args.eval_file_type
    save_result_path = args.save_result_path
    iou_th = args.iou_th

    eval_model = EVAL_MODEL(eval_data_dir, pre_data_dir, eval_file_type,
                            save_result_path, iou_th)
    eval_model.start_eval()
