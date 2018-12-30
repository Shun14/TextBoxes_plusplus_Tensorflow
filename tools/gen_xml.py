from __future__ import print_function
import os
from lxml import etree
import xml.dom.minidom
import sys
import random

import numpy as np
import codecs 
import cv2

def process_convert(txt_name, DIRECTORY_ANNOTATIONS, new_version=True):
    
    # Read the txt annotation file.
    filename = os.path.join(DIRECTORY_ANNOTATIONS, txt_name)
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    annotation_xml = xml.dom.minidom.Document()

    root = annotation_xml.createElement('annotation')
    annotation_xml.appendChild(root)
    
    nodeFolder = annotation_xml.createElement('folder')
    root.appendChild(nodeFolder)

    nodeFilename = annotation_xml.createElement('filename')
    nodeFilename.appendChild(annotation_xml.createTextNode(filename))
    root.appendChild(nodeFilename)

    img_name = filename[:-4]
    if cv2.imread(img_name) is not None:
        h, w, c = cv2.imread(img_name).shape
    else:
        raise KeyError('img_name error:', img_name)
    nodeSize = annotation_xml.createElement('size')
    nodeWidth = annotation_xml.createElement('width')
    nodeHeight = annotation_xml.createElement('height')
    nodeDepth = annotation_xml.createElement('depth')
    nodeWidth.appendChild(annotation_xml.createTextNode(str(w)))
    nodeHeight.appendChild(annotation_xml.createTextNode(str(h)))
    nodeDepth.appendChild(annotation_xml.createTextNode(str(c)))
    
    nodeSize.appendChild(nodeWidth)
    nodeSize.appendChild(nodeHeight)
    nodeSize.appendChild(nodeDepth)
    root.appendChild(nodeSize)    
    for l in lines:
        l = l.encode('utf-8').decode('utf-8-sig')
        l = l.strip().split(',')
        difficult = 0
        label = 1
        x1_text = ''
        x2_text = ''
        x3_text = ''
        x4_text = ''
        y1_text = ''
        y2_text = ''
        y3_text = ''
        y4_text = ''

        if new_version is True:
            label_name = str(l[-1])
            if label_name == 'none':
                difficult = 1
            else:
                difficult = 0
            label = label_name
            xmin_text = str(int(float(l[1])))
            ymin_text = str(int(float(l[2])))
            xmax_text = str(int(float(l[3])))
            ymax_text = str(int(float(l[4])))
            x1_text = xmin_text
            x2_text = xmax_text
            x3_text = xmax_text
            x4_text = xmin_text

            y1_text = ymin_text
            y2_text = ymin_text
            y3_text = ymax_text
            y4_text = ymax_text
        else:
            print(l)
            xs = [ int(l[i])  for i in (0, 2, 4, 6)]
            ys = [ int(l[i]) for i in (1, 3, 5, 7)]
            xmin_text = str(min(xs))
            ymin_text = str(min(ys))
            xmax_text = str(max(xs))
            ymax_text = str(max(ys))
            x1_text = str(xs[0])
            x2_text = str(xs[1])
            x3_text = str(xs[2])
            x4_text = str(xs[3])

            y1_text = str(ys[0])
            y2_text = str(ys[1])
            y3_text = str(ys[2])
            y4_text = str(ys[3])


        nodeObject = annotation_xml.createElement('object')
        nodeDifficult = annotation_xml.createElement('difficult')
        nodeDifficult.appendChild(annotation_xml.createTextNode(str(difficult)))
        nodeName = annotation_xml.createElement('name')
        nodeName.appendChild(annotation_xml.createTextNode(str(label)))

        nodeBndbox = annotation_xml.createElement('bndbox')    
        nodexmin = annotation_xml.createElement('xmin')
        nodexmin.appendChild(annotation_xml.createTextNode(xmin_text))
        nodeymin = annotation_xml.createElement('ymin')
        nodeymin.appendChild(annotation_xml.createTextNode(ymin_text))
        nodexmax = annotation_xml.createElement('xmax')
        nodexmax.appendChild(annotation_xml.createTextNode(xmax_text))
        nodeymax = annotation_xml.createElement('ymax')
        nodeymax.appendChild(annotation_xml.createTextNode(ymax_text))

        nodex1 = annotation_xml.createElement('x1')
        nodex1.appendChild(annotation_xml.createTextNode(x1_text))
        nodex2 = annotation_xml.createElement('x2')
        nodex2.appendChild(annotation_xml.createTextNode(x2_text))
        nodex3 = annotation_xml.createElement('x3')
        nodex3.appendChild(annotation_xml.createTextNode(x3_text))
        nodex4 = annotation_xml.createElement('x4')
        nodex4.appendChild(annotation_xml.createTextNode(x4_text))


        nodey1 = annotation_xml.createElement('y1')
        nodey1.appendChild(annotation_xml.createTextNode(y1_text))

        nodey2 = annotation_xml.createElement('y2')
        nodey2.appendChild(annotation_xml.createTextNode(y2_text))

        nodey3 = annotation_xml.createElement('y3')
        nodey3.appendChild(annotation_xml.createTextNode(y3_text))
        nodey4 = annotation_xml.createElement('y4')
        nodey4.appendChild(annotation_xml.createTextNode(y4_text))

        nodeBndbox.appendChild(nodexmin)
        nodeBndbox.appendChild(nodeymin)
        nodeBndbox.appendChild(nodexmax)
        nodeBndbox.appendChild(nodeymax)
        nodeBndbox.appendChild(nodex1)
        nodeBndbox.appendChild(nodex2)
        nodeBndbox.appendChild(nodex3)
        nodeBndbox.appendChild(nodex4)

        nodeBndbox.appendChild(nodey1)
        nodeBndbox.appendChild(nodey2)
        nodeBndbox.appendChild(nodey3)
        nodeBndbox.appendChild(nodey4)

        nodeObject.appendChild(nodeDifficult)
        nodeObject.appendChild(nodeName)
        nodeObject.appendChild(nodeBndbox)
        root.appendChild(nodeObject)
    

    
    xml_path = os.path.join(DIRECTORY_ANNOTATIONS, txt_name[0:-4] + '.xml')
    fp = open(xml_path, 'w')
    annotation_xml.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

    

def get_all_txt(directory, new_version=False):
    count = 0
    for root,dirs,files in os.walk(directory):
        for each in files:
            if each.split('.')[-1] == 'txt':
                count += 1
                print(count, each)
                
                process_convert(each, root, new_version)
                


if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='icdar15 generate xml tools')
    parser.add_argument('--in_dir','-i', default='/home/zsz/datasets/icdar15_anno/eval_img_2018.9.25/icdar15_eval_final', type=str)
    args = parser.parse_args()
    directory = args.in_dir
    get_all_txt(directory, False)
