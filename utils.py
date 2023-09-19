# import torch
# import os, sys, json, argparse
# import random
# from random import choice, sample
# from tensorboardX import SummaryWriter
# from Group_API import person_analysis,group_analysis
from tqdm import tqdm
import numpy as np
# generate data for training:
# 1. generate video with corresbonding(scene,personidA,personidB,t) from raw data
# 2. save video to pos,neg folder
# 3. script to clean data, disable dirty sample in overall dataset
# 4. make clean dataset to train and test

# cherry pick on the same time: find 2 tranjactory similarity is high, but when zoom in perceptual distance is low
# (no interaction)

import os
import cv2

# generate data for training:
# 1. generate video with corresbonding(scene,personidA,personidB,t) from raw data
# 2. save video to pos,neg folder
# 3. script to clean data, disable dirty sample in overall dataset
# 4. make clean dataset to train and test

# cherry pick on the same time: find 2 trajectory similarity is high, but when zoom in perceptual distance is low
# (no interaction)

from Group_API import SCENE_NAMES, IMG_LOAD_ROOT_PATH_30
from Group_API import img_urls_get, person_analysis, group_analysis
# Define constants
ANNO_ROOT_DIR = "<FILL_WITH_RIGHT_PATH>"
CONFIDENCE = ('High', 'Middle', 'Low')
ABOVE_PROPORTION = ('normal', 'hide', 'serious_hide', 'disappear')


def _restrain_img_range(x1y1x2y2, max_height, max_width):
    if x1y1x2y2[0] < 0:
        x1y1x2y2[0] = 0
    elif x1y1x2y2[0] > max_width:
        x1y1x2y2[0] = max_width
    if x1y1x2y2[1] < 0:
        x1y1x2y2[1] = 0
    elif x1y1x2y2[1] > max_height:
        x1y1x2y2[1] = max_width
    if x1y1x2y2[2] < 0:
        x1y1x2y2[2] = 0
    elif x1y1x2y2[2] > max_width:
        x1y1x2y2[2] = max_width
    if x1y1x2y2[3] < 0:
        x1y1x2y2[3] = 0
    elif x1y1x2y2[3] > max_height:
        x1y1x2y2[3] = max_height

    return x1y1x2y2

def _check_above_proportion(input_above_proportion, criterion):
    if input_above_proportion in ABOVE_PROPORTION[:criterion]:
        return True
    else:
        return False


def _check_confidence(input_confidence, criterion):
    if input_confidence in CONFIDENCE[:criterion]:
        return True
    else:
        return False


def _restrain_between_0_1(values_list):
    return_list = []
    for value in values_list:
        if value < 0:
            new_value = 0
        elif value > 1:
            new_value = 1
        else:
            new_value = value
        return_list.append(new_value)

    return return_list


def _move_box_center(x_c, y_c, margin, src_aspect_ratio, aspect_ratio):
    if x_c - margin < 0:
        diff = margin - x_c
        x_c += diff
    if x_c + margin > 1:
        diff = x_c + margin - 1
        x_c -= diff
    if y_c - margin * src_aspect_ratio / aspect_ratio < 0:
        diff = margin * src_aspect_ratio / aspect_ratio - y_c
        y_c += diff
    if y_c + margin * src_aspect_ratio / aspect_ratio > 1:
        diff = y_c + margin * src_aspect_ratio / aspect_ratio - 1
        y_c -= diff

    return x_c, y_c


def _cal_bbox_dist(bbox1, bbox2):
    """
    :param bbox1: (x_tl,y_tl,x_br,y_br)
    :param bbox2: (x_tl,y_tl,x_br,y_br)
    :return: 计算两个矩形的距�?如果相交返回-1 否则返回两矩形距�?    计算过程�?    1. 以其中一个矩形的的左下点为原点做一�?维坐标系，判断另外一个矩形的左下点在这个坐标系的哪个象限（其他点也可以，但是被比较的也得是相同的点）�?    2. 根据所在不同的象限分别在矩形上取不同的一个点作为比较�?    3. 对取到的两个点做向量的减法运算；
    4. 通过判断的相减后的向量来的到两个矩形的距�?    """

    if bbox1[1] < bbox2[1]:
        is_inversion = bbox1[0] < bbox2[0]
        if is_inversion:
            point1_x, point1_y = bbox1[2], bbox1[3]
            point2_x, point2_y = bbox2[0], bbox2[1]
        else:
            point1_x, point1_y = bbox1[0], bbox1[3]
            point2_x, point2_y = bbox2[2], bbox2[1]
    else:
        is_inversion = bbox1[0] > bbox2[0]
        if is_inversion:
            point1_x, point1_y = bbox2[2], bbox2[3]
            point2_x, point2_y = bbox1[0], bbox1[1]
        else:
            point1_x, point1_y = bbox2[0], bbox2[3]
            point2_x, point2_y = bbox1[2], bbox1[1]

    d_point_x, d_point_y = point2_x - point1_x, point2_y - point1_y
    if not is_inversion:
        d_point_x = -d_point_x
    if d_point_x < 0 and d_point_y < 0:
        return -1
    if d_point_x < 0:
        return d_point_y
    if d_point_y < 0:
        return d_point_x
    return (d_point_x ** 2 + d_point_y ** 2) ** 0.5


def get_interaction_seq_pos(src_height,
                            src_width,
                            person_info,
                            group_info,
                            person_id_A,
                            person_id_B,
                            video_quality=1,
                            interaction_confidence=1,
                            above_proportion=1,
                            context_margin=0.6,
                            distance_thres=4,
                            aspect_ratio=1):
    """
    :param scene_id: int;
    :param person_id_A: person ID of A int;
    :param person_id_B: person ID of B int;
    :param video_quality: 1 is 'High' ; 2 is 'Middle' and above; 3 is 'Low' and above;
    :param interaction_confidence: 1 is 'High' ; 2 is 'Middle' and above; 3 is 'Low' and above;
    :param above_proportion: the threshold of occlusion ratio, 1 is 'normal', 2 is 'hide' and above, 3 is 'serious_hide'
    and above, 4 is 'disappear' and above;
    :param context_margin: the parameter to control context area, please set more than 0.5;
    :param distance_thres: the threshold of distance between 2 person;
    :param aspect_ratio: the aspect ratio of box containing 2 person

    :return:
        case I: if two person are not in the same group, return []
        case II: if two person in the same group , but have no high confidence interection return []
        case III: if same group and high confidence available, return as follow:
            a list of following content(dim:[number of interaction clip, number of one clip time seq bounding box])
        1. bbox list[(x,y,h,w,t),(x,y,h,w,t)]: bounding box 包含了两个人的bounding box,并且两人之间的距离不超过distance_thres个两个人bounding
        box的最大宽�? bounding box长宽比恒定但像素大小不固�?
        2. bbox list[(x,y,h,w,t),(x,y,h,w,t)]:bounding box 包含了A的bounding box
        3. bbox list[(x,y,h,w,t),(x,y,h,w,t)]:bounding box 包含了B的bounding box
    """
    # scene_name = SCENE_NAMES[scene_id]
    # groups_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'groups.json')
    # print("reading json from {}".format(groups_anno_path))
    # groups_list = group_analysis(groups_anno_path)
    groups_list = group_info
    assert isinstance(person_id_A,int)
    assert isinstance(person_id_B,int)
    # find interaction clips of 2 candidate person from groups annotation
    interaction_clips = []
    for group_dict in groups_list:
        if group_dict['group_type'] != 'Single person':
            group_members = [person_dict['idx'] for person_dict in group_dict['person']]
            assert isinstance(group_members[0],int)
            if person_id_A in group_members and person_id_B in group_members:
                if _check_confidence(group_dict['video_quality'], video_quality):
                    for interact_dict in group_dict['interact_states']:
                        interact_members = interact_dict['members']
                        if _check_confidence(interact_dict['interact_confidence'], interaction_confidence):
                            if person_id_A in interact_members and person_id_B in interact_members:
                                interaction_clips.append({
                                    "state_cate": interact_dict['state_cate'],
                                    "start_frame": interact_dict['start_frame'],
                                    "end_frame": interact_dict['end_frame']
                                })

    # print('#interaction: ', len(interaction_clips))
    if not interaction_clips:
        return []

    # persons_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'persons.json')
    # persons_list = person_analysis(persons_anno_path)
    persons_list = person_info
    # img_urls = img_urls_get(persons_anno_path)
    # src_img_url = img_urls[0]
    # src_img_path = os.path.join(IMG_LOAD_ROOT_PATH_30, scene_name,"IMG", src_img_url)
    # src_img = cv2.imread(src_img_path)
    # src_height, src_width = src_img.shape[:2]
    src_aspect_ratio = src_width / src_height
    person_rects_A, person_rects_B = {}, {}
    person_occls_A, person_occls_B = {}, {}

    # find location of 2 candidate person from persons annotation
    # when the distance of 2 person is more than 4 mean box width, cut or split clips
    # when both of 2 person is occluded, cut or split clips
    # aspect ratio (width : height) can be set for the box containing 2 person
    # context margin can be set for the box containing 2 person
    for person_dict in persons_list:
        if person_dict['idx'] == person_id_A:
            for frame_dict in person_dict['frames list']:
                person_rects_A[frame_dict['frame_id']] = frame_dict['rect']
                person_occls_A[frame_dict['frame_id']] = person_dict['above proportion'][frame_dict['frame_id']]
        if person_dict['idx'] == person_id_B:
            for frame_dict in person_dict['frames list']:
                person_rects_B[frame_dict['frame_id']] = frame_dict['rect']
                person_occls_B[frame_dict['frame_id']] = person_dict['above proportion'][frame_dict['frame_id']]
    if not person_rects_A or not person_rects_B:
        return []

    return_list = []
    two_person_frames, A_frames, B_frames = [], [], []
    for interaction_clip_dict in interaction_clips:
        if two_person_frames:
            return_list.append([two_person_frames, A_frames, B_frames])
            two_person_frames, A_frames, B_frames = [], [], []
        start_frame = max(interaction_clip_dict['start_frame'], min(list(person_rects_A.keys())), min(list(person_rects_B.keys())))
        end_frame = min(interaction_clip_dict['end_frame'], max(list(person_rects_A.keys())), max(list(person_rects_B.keys())))
        if start_frame < end_frame:
            for i in range(start_frame, end_frame + 1):
                A_x_tl, A_y_tl, A_x_br, A_y_br = person_rects_A[i]['tl']['x'], person_rects_A[i]['tl']['y'], \
                                                 person_rects_A[i]['br']['x'], person_rects_A[i]['br']['y']
                B_x_tl, B_y_tl, B_x_br, B_y_br = person_rects_B[i]['tl']['x'], person_rects_B[i]['tl']['y'], \
                                                 person_rects_B[i]['br']['x'], person_rects_B[i]['br']['y']
                bbox_dist = _cal_bbox_dist((A_x_tl, A_y_tl, A_x_br, A_y_br), (B_x_tl, B_y_tl, B_x_br, B_y_br))
                if (_check_above_proportion(person_occls_A[i], above_proportion) or _check_above_proportion(
                        person_occls_B[i], above_proportion)) and bbox_dist < distance_thres * (A_x_br - A_x_tl +
                                                                                                B_x_br - B_x_tl) / 2:
                    A_frames.append((A_x_tl, A_y_tl, A_x_br, A_y_br, i))
                    B_frames.append((B_x_tl, B_y_tl, B_x_br, B_y_br, i))
                    x_min, y_min = min(A_x_tl, B_x_tl), min(A_y_tl, B_y_tl)
                    x_max, y_max = max(A_x_br, B_x_br), max(A_y_br, B_y_br)
                    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
                    margin = max(x_max - x_min, y_max - y_min) * context_margin
                    x_c, y_c = _move_box_center(x_c, y_c, margin, src_aspect_ratio, aspect_ratio)
                    two_person_frames.append((x_c - margin, y_c - margin * src_aspect_ratio / aspect_ratio,
                                              x_c + margin, y_c + margin * src_aspect_ratio / aspect_ratio, i))
                else:
                    if two_person_frames:
                        return_list.append([two_person_frames, A_frames, B_frames])
                        two_person_frames, A_frames, B_frames = [], [], []

    if two_person_frames:
        return_list.append([two_person_frames, A_frames, B_frames])
    return return_list


def get_interaction_seq_neg(src_img,
                            person_info,
                            group_info,
                            # scene_id,
                            person_id_A,
                            person_id_B,
                            video_quality=1,
                            above_proportion=1,
                            context_margin=0.55,
                            distance_thres=4,
                            aspect_ratio=1
                            ):
    """
    :param scene_id;
    :param person_id_A: person ID of A;
    :param person_id_B: person ID of B;
    :param above_proportion: the threshold of occlusion ratio, 1 is 'normal', 2 is 'hide' and above, 3 is 'serious_hide'
    and above, 4 is 'disappear' and above;
    :param context_margin: the parameter to control context area, please set more than 0.5;
    :param distance_thres: the threshold of distance between 2 person;
    :param aspect_ratio: the aspect ratio of box containing 2 person
    :return:
        case I: if two person are not in the same group and have frames seq that satisfy the following:
            两人之间的距离不超过6个两个人bounding box的宽度最大�?
        case II: if two person are in the same group and have frames seq that satisfy the following:
            两人之间没有互动且距离不超过6个两个人bounding box的宽度最大�?
            return: a list of following content(dim:[number of interaction clip, number of one clip time seq bounding box])
            1. bbox list[(x,y,h,w,t),(x,y,h,w,t)]: bounding box 包含了两个人的bounding box,并且两人之间的距离不超过distance_thres个两个人bounding
            box的最大宽�? bounding box长宽比恒定但像素大小不固�?
            2. bbox list[(x,y,h,w,t),(x,y,h,w,t)]:bounding box 包含了A的bounding box
            3. bbox list[(x,y,h,w,t),(x,y,h,w,t)]:bounding box 包含了B的bounding box
        case III: return []
    """
    # scene_name = SCENE_NAMES[scene_id]
    # groups_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'groups.json')
    # groups_list = group_analysis(groups_anno_path)
    groups_list = group_info

    # find interaction clips of 2 candidate person from groups annotation
    interaction_clips = []
    assert isinstance(person_id_A,int)
    assert isinstance(person_id_B,int)
    for group_dict in groups_list:
        if group_dict['group_type'] != 'Single person':
            group_members = [person_dict['idx'] for person_dict in group_dict['person']]
            assert isinstance(group_members[0],int)
            if person_id_A in group_members and person_id_B in group_members:
                if _check_confidence(group_dict['video_quality'], video_quality):
                    for interact_dict in group_dict['interact_states']:
                        interact_members = interact_dict['members']
                        if person_id_A in interact_members and person_id_B in interact_members:
                            interaction_clips.append({
                                "state_cate": interact_dict['state_cate'],
                                "start_frame": interact_dict['start_frame'],
                                "end_frame": interact_dict['end_frame']
                            })

    # print('#interaction: ', len(interaction_clips))

    # persons_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'persons.json')
    # persons_list = person_analysis(persons_anno_path)
    persons_list = person_info
    # img_urls = img_urls_get(persons_anno_path)
    # src_img_url = img_urls[0]
    # src_img_path = os.path.join(IMG_LOAD_ROOT_PATH_30, scene_name,"IMG", src_img_url)
    # src_img = cv2.imread(src_img_path)
    src_height, src_width = src_img.shape[:2]
    src_aspect_ratio = src_width / src_height
    person_rects_A, person_rects_B = {}, {}
    person_occls_A, person_occls_B = {}, {}

    # find location of 2 candidate person from persons annotation
    # when the distance of 2 person is more than 4 mean box width, cut or split clips
    # when both of 2 person is occluded, cut or split clips
    # when 2 person is in interaction, cut or split clips
    # aspect ratio (width : height) can be set for the box containing 2 person
    # context margin can be set for the box containing 2 person
    for person_dict in persons_list:
        if person_dict['idx'] == person_id_A:
            for frame_dict in person_dict['frames list']:
                person_rects_A[frame_dict['frame_id']] = frame_dict['rect']
                person_occls_A[frame_dict['frame_id']] = person_dict['above proportion'][frame_dict['frame_id']]
        if person_dict['idx'] == person_id_B:
            for frame_dict in person_dict['frames list']:
                person_rects_B[frame_dict['frame_id']] = frame_dict['rect']
                person_occls_B[frame_dict['frame_id']] = person_dict['above proportion'][frame_dict['frame_id']]
    if not person_rects_A or not person_rects_B:
        return []

    return_list = []
    two_person_frames, A_frames, B_frames = [], [], []
    start_frame = max(min(list(person_rects_A.keys())), min(list(person_rects_B.keys())))
    end_frame = min(max(list(person_rects_A.keys())), max(list(person_rects_B.keys())))

    if start_frame < end_frame:
        for i in range(start_frame, end_frame + 1):
            interaction_flag = False
            for interaction_clip in interaction_clips:
                if interaction_clip['start_frame'] <= i <= interaction_clip['end_frame']:
                    interaction_flag = True
            A_x_tl, A_y_tl, A_x_br, A_y_br = person_rects_A[i]['tl']['x'], person_rects_A[i]['tl']['y'], \
                                             person_rects_A[i]['br']['x'], person_rects_A[i]['br']['y']
            B_x_tl, B_y_tl, B_x_br, B_y_br = person_rects_B[i]['tl']['x'], person_rects_B[i]['tl']['y'], \
                                             person_rects_B[i]['br']['x'], person_rects_B[i]['br']['y']
            bbox_dist = _cal_bbox_dist((A_x_tl, A_y_tl, A_x_br, A_y_br), (B_x_tl, B_y_tl, B_x_br, B_y_br))

            if (_check_above_proportion(person_occls_A[i], above_proportion) or _check_above_proportion(
                    person_occls_B[i], above_proportion)) and bbox_dist < distance_thres * (A_x_br - A_x_tl +
                                                                    B_x_br - B_x_tl) / 2 and not interaction_flag:

                A_frames.append((A_x_tl, A_y_tl, A_x_br, A_y_br, i))
                B_frames.append((B_x_tl, B_y_tl, B_x_br, B_y_br, i))
                x_min, y_min = min(A_x_tl, B_x_tl), min(A_y_tl, B_y_tl)
                x_max, y_max = max(A_x_br, B_x_br), max(A_y_br, B_y_br)
                x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
                margin = max(x_max - x_min, y_max - y_min) * context_margin
                x_c, y_c = _move_box_center(x_c, y_c, margin, src_aspect_ratio, aspect_ratio)
                two_person_frames.append((x_c - margin, y_c - margin * src_aspect_ratio / aspect_ratio,
                                          x_c + margin, y_c + margin * src_aspect_ratio / aspect_ratio, i))
            else:
                if two_person_frames:
                    return_list.append([two_person_frames, A_frames, B_frames])
                    two_person_frames, A_frames, B_frames = [], [], []

    if two_person_frames:
        return_list.append([two_person_frames, A_frames, B_frames])
    return return_list


def get_person_traj_group_list(person_json_path,group_json_path):
    rt=[]
    persons_file=person_analysis(person_json_path)
    group_file=group_analysis(group_json_path)
    max_frame_id=0

    for person in persons_file:
        for frame in person["frames list"]:
            max_frame_id=max(max_frame_id,frame['frame_id'])

    for person in persons_file:
        tmp_person = {}
        tmp_person["traj"] = []
        tmp_person["group"]=[]
        tmp_person["same_group_persons"]=[]
        tmp_person["diff_group"]=[]
        tmp_person["different_group_persons"]=[]
        for frame_id in range(len(person["frames list"])):
            tmp_person["traj"].append(
                [
                    person["frames list"][frame_id]['frame_id']/float(max_frame_id),
                    person["frames list"][frame_id]['rect']['tl']['y'],
                    person["frames list"][frame_id]['rect']['tl']['x'],
                    person["frames list"][frame_id]['rect']['br']['y'],
                    person["frames list"][frame_id]['rect']['br']['x'],
                    person["face orientation"][frame_id]
                ]
            )
        rt.append(tmp_person)
    for group in group_file:
        if group['group_type']!='Single person':
            for group_member in group["person"]:
                person_id=group_member["idx"]
                rt[person_id]["group"].append(group["group_id"])
    for id,person in enumerate(rt,0):

        for pos_group in person["group"]:
            for positive_person in group_file[int(pos_group)]["person"]:
                rt[id]["same_group_persons"].append(positive_person['idx'])

        for neg_group in group_file:
            if neg_group["group_id"] in person["group"]:
                continue
            tmp_neg_group={
                "group_id":neg_group["group_id"],
                "person_id":[]
            }
            for negtive_person in group_file[int(neg_group["group_id"])]["person"]:
                if negtive_person['idx'] not in rt[id]["same_group_persons"]:
                    tmp_neg_group["person_id"].append(negtive_person['idx'])
            rt[id]["diff_group"].append(tmp_neg_group)

    return (rt,persons_file,group_file)


def get_video_from_scene_with_mask(scene_name, clips_bbox_list, resize=(500, 500)):
    """
    :param scene_id:
    :param clips_bbox_list: [[A, B, C] * #clip]
            A: bbox list[(x_tl, y_tl, x_br, y_br, t),(x_tl, y_tl, x_br, y_br, t)]: bounding box 包含了两个人的bounding box
            B: bbox list[(x_tl, y_tl, x_br, y_br, t),(x_tl, y_tl, x_br, y_br, t)]:bounding box 包含了A的bounding box
            C: bbox list[(x_tl, y_tl, x_br, y_br, t),(x_tl, y_tl, x_br, y_br, t)]:bounding box 包含了B的bounding box
    :param resize: (H, W)最后输出的图像resize在的大小
    :return: [clips_num, frame_num, H, W, 4] (the last channel of image is mask, 255 means bbox, 0 means background)
    """
    # scene_name = SCENE_NAMES[scene_id]
    persons_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'persons.json')
    img_urls = img_urls_get(persons_anno_path)

    return_list = [[] for i in range(len(clips_bbox_list))]

    for img_id in tqdm(range(len(img_urls))):
        # if img_id>50:
        #     break
        src_img_loaded = False
        src_img,src_height, src_width=None,None,None
        for clip_id, bbox_list in enumerate(clips_bbox_list):
            for loc, A in enumerate(bbox_list[0]):
                frame_id = A[4]
                if img_id == frame_id:
                    if not src_img_loaded:
                        img_url = img_urls[frame_id]
                        img_path = os.path.join(IMG_LOAD_ROOT_PATH_30, scene_name,"IMG", img_url)
                        src_img = cv2.imread(img_path)
                        src_height, src_width = src_img.shape[:2]
                        src_img_loaded = True
                    B, C = bbox_list[1][loc], bbox_list[2][loc]
                    bbox_A = _restrain_between_0_1(A[0:4])
                    bbox_B = _restrain_between_0_1(B[0:4])
                    bbox_C = _restrain_between_0_1(C[0:4])

                    x_tl, y_tl, x_br, y_br = bbox_A[0], bbox_A[1], bbox_A[2], bbox_A[3]
                    m1x_tl, m1y_tl, m1x_br, m1y_br = round((bbox_B[0] - x_tl) * src_width), \
                                                     round((bbox_B[1] - y_tl) * src_height), \
                                                     round((bbox_B[2] - x_tl) * src_width), \
                                                     round((bbox_B[3] - y_tl) * src_height)
                    m2x_tl, m2y_tl, m2x_br, m2y_br = round((bbox_C[0] - x_tl) * src_width), \
                                                     round((bbox_C[1] - y_tl) * src_height), \
                                                     round((bbox_C[2] - x_tl) * src_width), \
                                                     round((bbox_C[3] - y_tl) * src_height)

                    img = src_img[round(y_tl * src_height):round(y_br * src_height),
                          round(x_tl * src_width):round(x_br * src_width)]
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    m1x_tl, m1y_tl, m1x_br, m1y_br = _restrain_img_range([m1x_tl, m1y_tl, m1x_br, m1y_br], img.shape[0],
                                                                         img.shape[1])
                    m2x_tl, m2y_tl, m2x_br, m2y_br = _restrain_img_range([m2x_tl, m2y_tl, m2x_br, m2y_br], img.shape[0],
                                                                         img.shape[1])
                    mask[m1y_tl:m1y_br, m1x_tl:m1x_br] = np.ones((m1y_br - m1y_tl, m1x_br - m1x_tl),
                                                                 dtype=np.uint8) * 255
                    mask[m2y_tl:m2y_br, m2x_tl:m2x_br] = np.ones((m2y_br - m2y_tl, m2x_br - m2x_tl),
                                                                 dtype=np.uint8) * 255

                    img_resized = cv2.resize(img, resize)
                    mask_resized = cv2.resize(mask, resize)
                    mask_resized = mask_resized[:, :, np.newaxis]
                    fused_img = np.concatenate((img_resized, mask_resized), axis=2)
                    # print(fused_img.shape)
                    return_list[clip_id].append(np.copy(fused_img))
                    # print('img id: ', img_id)
                    # cv2.imshow('', img_resized)
                    # cv2.waitKey(0)
                    # cv2.imshow('', mask_resized)
                    # cv2.waitKey(0)

    return return_list



# if __name__ == '__main__':
#     get_person_traj_group_list("/data/anno_30/FPS30_清华视频-�?�?HIT_Canteen_frames-人物帧属性结果V1.4.json","/data/anno_30/FPS30_清华视频-�?�?HIT_Canteen_frames-第二步人物关系结果V1.4.json")
# main function for debugging
# if __name__ == '__main__':
#     scene_name = SCENE_NAMES[0]
#     groups_anno_path = os.path.join(ANNO_ROOT_DIR, scene_name, 'groups.json')
#     groups_list = group_analysis(groups_anno_path)
#     # for group_dict in groups_list:
#     #     if group_dict['group_type'] != 'Single person':
#     #         group_members = [person_dict['idx'] for person_dict in group_dict['person']]
#     #         if len(group_members) == 2:
#     #             interaction_seq_pos = get_interaction_seq_pos(1, group_members[0], group_members[1])
#     interaction_seq_pos = get_interaction_seq_pos(0, 144, 143)
#     print('#clips: ', len(interaction_seq_pos))
#     clips_bbox_list = []
#     for temp in interaction_seq_pos:
#         clips_bbox_list.append(temp[0])
#         # print(temp[0])

    # video_from_scene = get_video_from_scene(0, clips_bbox_list)