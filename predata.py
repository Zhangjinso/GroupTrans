import os
import cv2
import json
import random
import math
import copy
MAX_IMAGE_NUM = 300
# ''' The organization structure of persons_list:
#                 persons_list({
#                 'frames list': value_list(
#                     {'frame_id': 0, 
#                     'end': False, 
#                     'rect': {
#                         'tl': {'y': 0.9009288633, 'x': 0.042706294}, 
#                         'br': {'y': 1.1332216286, 'x': 0.0951287881}
#                         }
#                     }
#                 ),
#                 'face orientation': face_orientation,
#                 'above proportion': above_proportion,   #  'normal', 'hide', 'serious_hide', 'disappear'
#                 'idx': idx    # person id
#                 })
#         '''
#         persons_list.append({
#             'frames list': value_list,
#             'face orientation': face_orientation,
#             'above proportion': above_proportion,
#             'idx': idx
face_value ={'unsure':-1,'back':0.5*math.pi,'right back':0.25 * math.pi,'left back': 0.75 * math.pi,
'front':1.5 * math.pi,'right front':1.75 * math.pi,'left front':1.25 * math.pi,'left':math.pi,'right':0}
def person_analysis(person_gt_path):
    # person JSON reading and load info into memory
    with open(person_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
        persons_list = []
        for info in load_dict:# 每一个trackid（person idx）
            idx=info['track id']
            value_list = []
            face_orientation = []
            above_proportion = []
            for frame_info in info['frames']:
                tmp = {'frame_id':frame_info['frame id'] - 1,
                'rect':frame_info['rect']}
                value_list.append(tmp)
                if frame_info['face orientation'] =='':
                    face_orientation.append('-1')
                else:
                    face_orientation.append(face_value[frame_info['face orientation']])
                above_proportion.append(frame_info['occlusion'])
            persons_list.append({'frames list': value_list,
             'face orientation': face_orientation,
             'above proportion': above_proportion,
             'idx': idx-1})
    return persons_list
                
def group_analysis(group_gt_path):
    with open(group_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
    groups_list = load_dict
    for info in groups_list:
        for pp in info['person']:
            pp['idx']=pp['idx']-1
    return groups_list               

def jrdb_person_analysis(person_gt_path):
    # person JSON reading and load info into memory
    with open(person_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
        persons_list = []
        for info in load_dict:# 每一个trackid（person idx）
            idx=info['track id']
            value_list = []
            face_orientation = []
            above_proportion = []
            for frame_info in info['frames']:
                tmp = {'frame_id':frame_info['frame id'] - 1,
                'rect':frame_info['rect']}
                value_list.append(tmp)
                if frame_info['face orientation'] ==' ':
                    face_orientation.append('-1')
                else:
                    face_orientation.append(face_value[frame_info['face orientation']])
                above_proportion.append(frame_info['occlusion'])
            persons_list.append({'frames list': value_list,
             'face orientation': face_orientation,
             'above proportion': above_proportion,
             'idx': idx})
    return persons_list
                
def jrdb_group_analysis(group_gt_path):
    with open(group_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
    groups_list = load_dict
    # for info in groups_list:
    #     for pp in info['person']:
    #         pp['idx']=pp['idx']
    return groups_list 