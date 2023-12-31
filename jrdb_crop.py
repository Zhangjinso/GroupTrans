import json
from PIL import Image
import os
import os.path
import numpy as np
import cv2 
from tqdm import tqdm
from numpy import random
img_path='./dataset/JRDB/images/image_stitched_key/'
ann_path = './dataset/jrdb_annotation/video_annos'
save_path = './dataset/JRDB/bbox_jrdb/'

ann_list= '''tressider-2019-03-16_0
svl-meeting-gates-2-2019-04-08_1
svl-meeting-gates-2-2019-04-08_0
stlc-111-2019-04-19_0
packard-poster-session-2019-03-20_2
packard-poster-session-2019-03-20_1
packard-poster-session-2019-03-20_0
memorial-court-2019-03-16_0
jordan-hall-2019-04-22_0
huang-lane-2019-02-12_0
huang-basement-2019-01-25_0
hewlett-packard-intersection-2019-01-24_0
gates-to-clark-2019-02-28_1
gates-basement-elevators-2019-01-17_1
gates-159-group-meeting-2019-04-03_0
forbes-cafe-2019-01-22_0
cubberly-auditorium-2019-04-22_0
clark-center-intersection-2019-02-28_0
clark-center-2019-02-28_0
bytes-cafe-2019-02-07_0
clark-center-2019-02-28_1
gates-ai-lab-2019-02-08_0
huang-2-2019-01-25_0
meyer-green-2019-03-16_0
nvidia-aud-2019-04-18_0
tressider-2019-03-16_1
tressider-2019-04-26_2'''
ann_list = ann_list.split('\n')

for i in range(0,len(ann_list)):
    img_path1 = img_path + ann_list[i]
    img_list = os.listdir(img_path1)# the imgs in the sequence
    store_path = save_path+ann_list[i]
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
    for img_name in tqdm(img_list):
        tmp1 = img_name.split('_')
        tmp2 = tmp1[2].split('.')
        frame = int(tmp2[0])
        tmp_index = i
        img_pt = os.path.join(img_path1,img_name)
        # print(img_pt)
        img = cv2.imread(img_pt)
        height = 480
        width = 3760
        ann_tracks = ann_path + '/' + ann_list[tmp_index] +'/tracks_new.json'
        with open(ann_tracks, 'r', encoding='UTF-8') as load_f:
            tracks_annotation = json.load(load_f)
            for id in tracks_annotation:# every tracks
                for frames in id['frames']:# every frames in each track
                    if frames['frame id'] == frame:                     
                        idx = id['track id']
                        tl_y = max(int(frames['rect']['tl']['y']*height),0)
                        tl_x = max(int(frames['rect']['tl']['x']*width),0)
                        br_y = min(int(frames['rect']['br']['y']*height),height)
                        br_x = min(int(frames['rect']['br']['x']*width),width)
                        if (br_x<=0 or  tl_x>=width or tl_y>=height or br_y<=0 or br_x<=tl_x or br_y<=tl_y):
                            continue
                            
                        cropped = img[tl_y:br_y,tl_x:br_x]
                        cv2.imwrite(f'{save_path}{ann_list[tmp_index]}/{ann_list[tmp_index]}_frame_{frame}_{idx}.jpg',cropped)
                    
    
        
        