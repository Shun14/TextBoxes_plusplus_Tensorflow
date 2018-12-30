# -*- coding: utf-8 -*-
import os
import sys
import random
import time
import numpy as np
import codecs
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import SubElement


def process_convert(name, DIRECTORY_ANNOTATIONS, img_path, save_xml_path):
    # Read the XML annotation file.
    filename = os.path.join(DIRECTORY_ANNOTATIONS, name)
    try:
        tree = ET.parse(filename)
    except:
        print('error:', filename, ' not exist')
        return False

    root = tree.getroot()
    size = root.find('size')
    if size is None:
        img = cv2.imread(img_path)
        print('jpg_path', img_path, img.shape)
        shape = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])]
        # size = SubElement(root, 'size')

    elif size.find('height').text is None or size.find('width').text is None:
        img = cv2.imread(img_path)
        print('jpg_path height', img_path, img.shape)
        shape = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])]
    elif int(size.find('height').text) == 0 or int(
            size.find('width').text) == 0:

        img = cv2.imread(img_path)
        print('jpg_path zero', img_path, img.shape)
        shape = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])]
    else:
        shape = [
            int(size.find('height').text),
            int(size.find('width').text),
            int(size.find('depth').text)
        ]

    height = size.find('height')
    height.text = str(shape[0])
    width = size.find('width')
    width.text = str(shape[1])

    for obj in root.findall('object'):
        difficult = int(obj.find('difficult').text)
        content = obj.find('name').text
        content = content.replace('\t', '  ')

        #if int(difficult) == 1 and content == '&amp;*@HUST_special':
        '''
        这里代表HUST_vertical是text
        '''
        if difficult == 0 and content != '&*@HUST_special' and content != '&*HUST_shelter':
            label_name = 'text'
        else:
            label_name = 'none'

        bbox = obj.find('bndbox')
        if obj.find('content') is None:
            content_sub = SubElement(obj, 'content')
            content_sub.text = content
        else:
            obj.find('content').text = content
        
        name_ele = obj.find('name')
        name_ele.text = label_name

        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        x1 = xmin
        x2 = xmax
        x3 = xmax
        x4 = xmin

        y1 = ymin
        y2 = ymin
        y3 = ymax
        y4 = ymax

        if bbox.find('x1') is None:
            x1_sub = SubElement(bbox, 'x1')
            x1_sub.text = x1
            x2_sub = SubElement(bbox, 'x2')
            x2_sub.text = x2
            x3_sub = SubElement(bbox, 'x3')
            x3_sub.text = x3
            x4_sub = SubElement(bbox, 'x4')
            x4_sub.text = x4

            y1_sub = SubElement(bbox, 'y1')
            y1_sub.text = y1
            y2_sub = SubElement(bbox, 'y2')
            y2_sub.text = y2
            y3_sub = SubElement(bbox, 'y3')
            y3_sub.text = y3
            y4_sub = SubElement(bbox, 'y4')
            y4_sub.text = y4
        else:
            bbox.find('y1').text = ymin
            bbox.find('y2').text = ymin
            bbox.find('y3').text = ymax
            bbox.find('y4').text = ymax
    #print(save_xml_path)
    tree.write(save_xml_path)
    return True


def process_convert_txt(name, DIRECTORY_ANNOTATIONS):
    # Read the XML annotation file.
    filename = os.path.join(DIRECTORY_ANNOTATIONS, name)
    try:
        tree = ET.parse(filename)
    except:
        print('error:', filename, ' not exist')
        return
    root = tree.getroot()
    all_txt_line = []
    for obj in root.findall('object'):

        bbox = obj.find('bndbox')

        difficult = int(obj.find('difficult').text)
        content = obj.find('content')
        if content is not None:
            content = content.text
        else:
            content = 0
        if difficult == 1 and content == '&*@HUST_special':
            continue
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        x1 = xmin
        x2 = xmax
        x3 = xmax
        x4 = xmin

        y1 = ymin
        y2 = ymin
        y3 = ymax
        y4 = ymax

        all_txt_line.append('{} {} {} {} {} {} {} {}\n'.format(
            x1, y1, x2, y2, x3, y3, x4, y4))

    txt_name = os.path.join(DIRECTORY_ANNOTATIONS, name[:-4] + '.txt')
    with codecs.open(txt_name, 'w', encoding='utf-8') as f:
        f.writelines(all_txt_line)


def get_all_img(directory, split_flag, logs_dir, output_dir):
    count = 0
    ano_path_list = []
    img_path_list = []
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    for root, dirs, files in os.walk(directory):
        for each in files:
            if each.split('.')[-1] == 'xml':
                xml_path = os.path.join(root, each[:-4] + '.xml')
                img_path = os.path.join(root, each[:-4] + '.png')
                if os.path.exists(img_path) == False:
                    img_path = os.path.join(root, each[:-4] + '.PNG')
                test_png = cv2.imread(img_path)
                if test_png is None or os.path.exists(xml_path) == False:
                    continue
                if output_dir is not None:
                    sub_path = root[len(directory)+1:]
                    sub_path = os.path.join(output_dir, sub_path)
                    if not os.path.exists(sub_path):
                        os.makedirs(sub_path)
                    save_xml_path = os.path.join(sub_path, each[:-4] + '.xml')
                else:
                    save_xml_path = xml_path
                if process_convert(each, root, img_path, save_xml_path):
                    ano_path_list.append('{},{}\n'.format(
                        img_path,
                        save_xml_path))
                    img_path_list.append('{}\n'.format(
                        img_path))
                count += 1
                if count % 1000 == 0:
                    print(count, time.time() - start_time)
    save_to_text(img_path_list, ano_path_list, count, split_flag, logs_dir)
    print('all over:', count)
    print('time:', time.time() - start_time)


def save_to_text(img_path_list, ano_path_list, count, split_flag, logs_dir):
    if split_flag == 'yes':
        train_num = int(count / 10. * 9.)
    else:
        train_num = count
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    with codecs.open(
            os.path.join(logs_dir, 'train_xml.txt'), 'w',
            encoding='utf-8') as f_xml, codecs.open(
                os.path.join(logs_dir, 'train.txt'), 'w',
                encoding='utf-8') as f_txt:
        f_xml.writelines(ano_path_list[:train_num])
        f_txt.writelines(img_path_list[:train_num])

    with codecs.open(
            os.path.join(logs_dir, 'test_xml.txt'), 'w',
            encoding='utf-8') as f_xml, codecs.open(
                os.path.join(logs_dir, 'test.txt'), 'w',
                encoding='utf-8') as f_txt:
        f_xml.writelines(ano_path_list[train_num:])
        f_txt.writelines(img_path_list[train_num:])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='icdar15 generate xml tools for standard format')
    parser.add_argument(
        '--in_dir',
        '-i',
        default=
        '/home/zsz/datasets/icdar15_anno/annotated_data_3rd_8thv2_cut_resize_margin8',
        type=str)
    parser.add_argument('--split_flag', '-s', default='no', type=str)
    parser.add_argument('--save_logs', '-l', default='logs', type=str)
    parser.add_argument('--output_dir', '-o', default=None, type=str)

    args = parser.parse_args()
    directory = args.in_dir
    split_flag = args.split_flag
    logs_dir = args.save_logs
    output_dir = args.output_dir
    get_all_img(directory, split_flag, logs_dir, output_dir)
