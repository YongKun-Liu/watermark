#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import cv2
import numpy as np

#config
longEdge = 500
flag = False

def preprocess(foldername, suffix="_processed"):

    dest_folder = foldername + suffix
    processed=os.path.abspath(dest_folder)

    if not os.path.exists(processed):
        os.mkdir(dest_folder)
    else:
        os.system('rm %s/*'%(processed))
    filenames = os.listdir(foldername)
    size = 0 
    imgs = []
    for item in filenames:
        filename = os.path.join(foldername, item)
        img = cv2.imread(filename)
        if img is not None:
            height, width, _ = img.shape
            if max(height, width) > size:
                size = max(height, width)
            temp = {'name':item, 'img': img}
            imgs.append(temp)
    for img in imgs:
        filename = img['name']
        src = img['img']
        if flag == True:
            size = longEdge
        h, w, _ = src.shape
        if max(h, w) > size:
            continue
        dst = preprocess_image(src, size)
        cv2.imwrite(os.path.join(dest_folder, filename), dst)

def preprocess_image(img, size=500):
    height, width, _ = img.shape
    assert max(height, width) <= size
    top = (size-height)//2
    bottom = size-height-top
    left = (size-width)//2
    right = size-width-left
    final_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return final_img 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Format : %s <foldername>"%(sys.argv[0]))
    else:
        preprocess(sys.argv[1])
