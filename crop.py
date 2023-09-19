import json
from PIL import Image
import os
import os.path
import numpy as np
import cv2 
from tqdm import tqdm
from numpy import random
img_path0='./dataset/panda_video/train/'
ann_path = './dataset/panda_annotation/video_annos'
save_path = './dataset/panda_annotation/bbox_crop/'
ann_list = ['01_University_Canteen','02_OCT_Habour','03_Xili_Crossroad','04_Primary_School','05_Basketball_Court'
,'06_Xinzhongguan','07_University_Campus','08_Xili_Street_1','09_Xili_Street_2','10_Huaqiangbei']

for i in range(0,1):
    img_path = os.path.join(img_path0, ann_list[i])
    print(img_path)
    img_list = os.listdir(img_path)# the imgs in the sequence
    for img_name in tqdm(img_list):
        tmp1 = img_name.split('_')
        tmp2 = tmp1[2].split('.')
        frame = int(tmp2[0])
        tmp_index = i
        img_pt = img_path + '/'+img_name
        img = cv2.imread(img_pt)
        height = img.shape[0]
        width = img.shape[1]
        ann_tracks = ann_path + '/' + ann_list[tmp_index] +'/tracks_new.json'
        with open(ann_tracks, 'r', encoding='UTF-8') as load_f:
            tracks_annotation = json.load(load_f)
            for id in tracks_annotation:# every tracks
                for frames in id['frames']:# every frames in each track
                    if frames['frame id'] == frame:                     
                        idx = id['track id']
                        tl_y = max((frames['rect']['tl']['y']),0)
                        tl_x = max((frames['rect']['tl']['x']),0)
                        br_y = min((frames['rect']['br']['y']),1)
                        br_x = min((frames['rect']['br']['x']),1)
                        h = br_y - tl_y
                        w = br_x - tl_x
                        tl_y = max(int(frames['rect']['tl']['y']*height),0)
                        tl_x = max(int(frames['rect']['tl']['x']*width),0)
                        br_y = min(int(frames['rect']['br']['y']*height),height)
                        br_x = min(int(frames['rect']['br']['x']*width),width)
                        if (br_x<=0 or  tl_x>=width or tl_y>=height or br_y<=0 or br_x<=tl_x or br_y<=tl_y):
                            continue
                            
                        cropped = img[tl_y:br_y,tl_x:br_x]
                        cv2.imwrite(f'{save_path}{ann_list[tmp_index]}/{ann_list[tmp_index]}_frame_{frame}_{idx}.jpg',cropped)
                    
    
        
        