import os
import cv2
import json
import random
import math


PERSON_GT_JSON_PATHS = [
    '<FILL_WITH_RIGHT_PATH>',

]

PERSON_GT_JSON_PATHS_6 = [
    '<FILL_WITH_RIGHT_PATH>',

]

PERSON_GT_JSON_PATHS_30 = [
    '<FILL_WITH_RIGHT_PATH>',
]

GROUP_GT_JSON_PATHS = [
    '<FILL_WITH_RIGHT_PATH>',
    '<FILL_WITH_RIGHT_PATH>',
]

GROUP_GT_JSON_PATHS_30 = [
    '<FILL_WITH_RIGHT_PATH>',
    '<FILL_WITH_RIGHT_PATH>'
]

SCENE_NAMES = ['<FILL_WITH_RIGHT_PATH>',
]

GT_JSON_ROOT_PATH = '<FILL_WITH_RIGHT_PATH>'
GT_JSON_ROOT_PATH_30 = "<FILL_WITH_RIGHT_PATH>"
IMG_LOAD_ROOT_PATH = '<FILL_WITH_RIGHT_PATH>'
IMG_LOAD_ROOT_PATH_6 = '<FILL_WITH_RIGHT_PATH>'
IMG_LOAD_ROOT_PATH_30 = "<FILL_WITH_RIGHT_PATH>"
LINE_LENGTH_COEF = 0.85
MAX_IMAGE_NUM = 3600

IMG_SAVE_ROOT_PATH = '<FILL_WITH_RIGHT_PATH>'

# 调整这个值控制保存图片的大小
IMAGE_VIS_WIDTH = 6400


# R, G, B是 [0, 255]. H 是[0, 360]. S, V 是 [0, 1].
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def img_urls_get(person_gt_path):
    with open(person_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)

    items_list = load_dict['items']
    items_dict = items_list[0]

    return items_dict['uris']


def person_analysis(person_gt_path):
    # person JSON reading and load info into memory
    with open(person_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
    items_list = load_dict['items']
    items_dict = items_list[0]
    data_dict = items_dict['data']
    tracks_list = data_dict['tracks']
    persons_list = []
    for track_dict in tracks_list:
        frames_dict = track_dict['frames']
        if 'idx' in track_dict.keys():
            idx = track_dict['idx']
        else:
            raise KeyError(track_dict['attrs'])
        value_list = frames_dict['value']
        attrs_dict = frames_dict['attrs']
        above_proportion_dict = attrs_dict['above_proportion']
        face_orientation_dict = attrs_dict[' face_orientation']
        above_proportion = ['' for i in range(MAX_IMAGE_NUM)]
        face_orientation = [0 for i in range(MAX_IMAGE_NUM)]

        # analyse cover proportion
        if " normal" in above_proportion_dict.keys():
            for item in above_proportion_dict[" normal"]:
                for n in range(item[0], item[1] + 1):
                    above_proportion[n] = 'normal'
        if "hide" in above_proportion_dict.keys():
            for item in above_proportion_dict["hide"]:
                for n in range(item[0], item[1] + 1):
                    above_proportion[n] = 'hide'
        if "serious_hide" in above_proportion_dict.keys():
            for item in above_proportion_dict["serious_hide"]:
                for n in range(item[0], item[1] + 1):
                    above_proportion[n] = 'serious_hide'
        if " disappear" in above_proportion_dict.keys():
            for item in above_proportion_dict[" disappear"]:
                for n in range(item[0], item[1] + 1):
                    above_proportion[n] = 'disappear'
        # analyse face
        if "unsure" in face_orientation_dict.keys():
            for item in face_orientation_dict["unsure"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = -1
        if "back" in face_orientation_dict.keys():
            for item in face_orientation_dict["back"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 0.5 * math.pi
        if "right_back" in face_orientation_dict.keys():
            for item in face_orientation_dict["right_back"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 0.25 * math.pi
        if "left_back" in face_orientation_dict.keys():
            for item in face_orientation_dict["left_back"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 0.75 * math.pi
        if " front" in face_orientation_dict.keys():
            for item in face_orientation_dict[" front"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 1.5 * math.pi
        if "right_front" in face_orientation_dict.keys():
            for item in face_orientation_dict["right_front"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 1.75 * math.pi
        if "left_front" in face_orientation_dict.keys():
            for item in face_orientation_dict["left_front"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 1.25 * math.pi
        if "left" in face_orientation_dict.keys():
            for item in face_orientation_dict["left"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = math.pi
        if "right" in face_orientation_dict.keys():
            for item in face_orientation_dict["right"]:
                for n in range(item[0], item[1] + 1):
                    face_orientation[n] = 0

        ''' The organization structure of persons_list:
                persons_list({
                'frames list': value_list(
                    {'frame_id': 0, 
                    'end': False, 
                    'rect': {
                        'tl': {'y': 0.9009288633, 'x': 0.042706294}, 
                        'br': {'y': 1.1332216286, 'x': 0.0951287881}
                        }
                    }
                ),
                'face orientation': face_orientation,
                'above proportion': above_proportion,   #  'normal', 'hide', 'serious_hide', 'disappear'
                'idx': idx    # person id
                })
        '''
        persons_list.append({
            'frames list': value_list,
            'face orientation': face_orientation,
            'above proportion': above_proportion,
            'idx': idx
        })

    return persons_list


def group_analysis(group_gt_path):
    """
    The organization structure of groups_list's element:
    (1) w/o interaction:
    {
      "group_id": "0",
      "group_type": "single person",
      "person": [
        {
          "idx": 3,
          "gender": "male",
          "age": "middle_young_age"
        },
        {
          "idx": 4,
          "gender": "male",
          "age": "middle_young_age"
        }
      ],
      "reliability": "null"
    }

    (2) with interactions:
    {
      "group_id": "0",
      "group_type": "Family",
      "video_quality": "Middle",
      "reliability": "High",
      "person": [
        {
          "idx": 3,
          "gender": "male",
          "age": "middle_young_age"
          "member_confidence": "High"
        },
        {
          "idx": 4,
          "gender": "male",
          "age": "middle_young_age"
          "member_confidence": "Middle"
        }
      ],
      "interact_states": [
        {
        "interact_confidence": "Low",
        "state_cate": "Talking",
        "end_frame": 692,
        "members": [
            2,
            3
        ],
        "start_frame": 0,
        "interact_id": 1
        }
      ],
    }
    """

    # group JSON reading and draw boxes
    with open(group_gt_path, 'r', encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
    groups_list = load_dict['groups']

    return groups_list


def tracking_draw(img_save_path, img_load_path, persons_list, image_num=None):
    # 创建保存路径
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    filenames = []
    for root, dirs, files in os.walk(img_load_path):
        for file in files:
            if file[-3:] == 'jpg':
                filenames.append(file)
    filenames.sort()

    images = []
    print('Start to load all images!')
    # 把视频标注需要用到的图片载入内存，方便在上面绘制框
    for img_id, img_name in enumerate(filenames):
        img_path = img_load_path + img_name
        # load img and resize
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        scale_ratio = IMAGE_VIS_WIDTH / width
        img_resized = cv2.resize(img, (int(width * scale_ratio), int(height * scale_ratio)))
        images.append(img_resized)
        print('Loaded image: ' + img_name.split('_')[-1])
        if image_num:
            if img_id + 1 == image_num:
                break
    print('Images load finished!')

    for person_dict in persons_list:
        frames = person_dict['frames list']
        face_orientation = person_dict['face orientation']
        above_proportion = person_dict['above proportion']
        person_id = person_dict['idx']

        # transfer color space from RGB to HSV to improve visual perception
        h = 30 * int(person_id) % 360
        s = 0.8 + random.random() * 0.2
        v = 0.35 + random.random() * 0.5
        r, g, b = hsv2rgb(h, s, v)

        for frame in frames:
            frame_id = frame['frame_id']
            x1 = int(frame['rect']['tl']['x'] * width * scale_ratio)
            y1 = int(frame['rect']['tl']['y'] * height * scale_ratio)
            x2 = int(frame['rect']['br']['x'] * width * scale_ratio)
            y2 = int(frame['rect']['br']['y'] * height * scale_ratio)
            if image_num is None or frame_id + 1 <= image_num:
                cv2.rectangle(images[frame_id], (x1, y1), (x2, y2), (b, g, r), 1)
                # show person id
                cv2.putText(images[frame_id], str(person_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (b, g, r), 1)

                # show face orientation line
                theta = face_orientation[frame_id]
                radius = x2 - x1
                src_x = int((x1 + x2) / 2)
                src_y = y1
                # 将人物朝向绘制为线段
                if theta != -1:
                    cv2.circle(images[frame_id], (src_x, src_y), 3, (b, g, r), thickness=-1)
                    cv2.line(images[frame_id], (src_x, src_y),
                             (src_x + int(radius * LINE_LENGTH_COEF * math.cos(theta)),
                              src_y + int(radius * LINE_LENGTH_COEF * math.sin(theta))),
                             (b, g, r), 1)
    # 保存绘制结果图片
    print('Start to save painted images!')
    for i, img in enumerate(images):
        cv2.imwrite(img_save_path + str(i).zfill(4) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        print('Saved image: ' + str(i) + '.jpg')


# show_type can be: group / group type / reliability / above proportion
# text_type can be: null / person id / group id
def group_draw(img_save_path, img_load_path, persons_list, groups_list, show_type='group', text_type='group id', show_single_person=True, image_num=None):
    # 创建保存路径
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    filenames = []
    for root, dirs, files in os.walk(img_load_path):
        for file in files:
            if file[-3:] == 'jpg':
                filenames.append(file)
    filenames.sort()

    images = []
    print('Start to load all images!')
    # 把视频标注需要用到的图片载入内存，方便在上面绘制框
    for img_id, img_name in enumerate(filenames):
        img_path = img_load_path + img_name
        # load img and resize
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        scale_ratio = IMAGE_VIS_WIDTH / width
        img_resized = cv2.resize(img, (int(width * scale_ratio), int(height * scale_ratio)))
        images.append(img_resized)
        print('Loaded image: ' + img_name.split('_')[-1])
        if image_num:
            if img_id + 1 == image_num:
                break
    print('Images load finished!')

    # draw for each group
    for group_dict in groups_list:
        group_id = group_dict['group_id']
        group_type = group_dict['group_type']
        members_list = group_dict['person']

        if show_type == 'group type':
            if group_type == "acquaintances":
                r, g, b = 0, 0, 255
            elif group_type == "family":
                r, g, b = 0, 255, 0
        elif show_type == 'reliability':
            reliability = group_dict['reliability']
            if reliability == "high":
                r, g, b = 0, 255, 0
            elif reliability == "middle":
                r, g, b = 0, 0, 255
            elif reliability == "low":
                r, g, b = 255, 0, 0
        elif show_type == 'group':
            # transfer color space from RGB to HSV to improve visual perception
            h = 30 * int(group_id) % 360
            s = 0.8 + random.random() * 0.2
            v = 0.35 + random.random() * 0.5
            r, g, b = hsv2rgb(h, s, v)

        if show_single_person or group_type != "single person":
            for member_dict in members_list:
                id = member_dict['idx']
                #role = member_dict['role']
                n = 0
                for person_dict in persons_list:
                    if person_dict['idx'] == id:
                        n += 1
                        frames = person_dict['frames list']
                        face_orientation = person_dict['face orientation']
                        above_proportion = person_dict['above proportion']
                        for frame in frames:
                            frame_id = frame['frame_id']

                            if show_type == 'above proportion':
                                if above_proportion[frame_id] == "hide":
                                    r, g, b = 0, 255, 0
                                elif above_proportion[frame_id] == "serious_hide":
                                    r, g, b = 0, 0, 255
                                elif above_proportion[frame_id] == "disappear":
                                    r, g, b = 255, 0, 0
                                else:
                                    r, g, b = 0, 0, 0

                            x1 = int(frame['rect']['tl']['x'] * width * scale_ratio)
                            y1 = int(frame['rect']['tl']['y'] * height * scale_ratio)
                            x2 = int(frame['rect']['br']['x'] * width * scale_ratio)
                            y2 = int(frame['rect']['br']['y'] * height * scale_ratio)
                            if image_num is None or frame_id + 1 <= image_num:
                                cv2.rectangle(images[frame_id], (x1, y1), (x2, y2), (b, g, r), 1)
                                # show text
                                if text_type == 'person id':
                                    cv2.putText(images[frame_id], str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                                (b, g, r), 1)
                                elif text_type == 'group id':
                                    cv2.putText(images[frame_id], str(group_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                                (b, g, r), 1)
                                '''
                                elif text_type == 'role':
                                    cv2.putText(images[frame_id], role, (x1, y1),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (b, g, r), 2)
                                '''
                                # show face line
                                theta = face_orientation[frame_id]
                                radius = x2 - x1
                                src_x = int((x1 + x2) / 2)
                                src_y = y1
                                # 将人物朝向绘制为线段
                                if theta != -1:
                                    cv2.circle(images[frame_id], (src_x, src_y), 3, (b, g, r), thickness=-1)
                                    cv2.line(images[frame_id], (src_x, src_y),
                                             (src_x + int(radius * LINE_LENGTH_COEF * math.cos(theta)),
                                              src_y + int(radius * LINE_LENGTH_COEF * math.sin(theta))),
                                             (b, g, r), 1)

    print('Start to save painted images!')
    # 保存绘制结果图片
    for i, img in enumerate(images):
        cv2.imwrite(img_save_path + str(i).zfill(4) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        print('Saved image: ' + str(i) + '.jpg')


# show_type can be: group / group type / reliability / above proportion
# text_type can be: null / person id / group id
def proposal_draw(img_save_path, img_load_path, persons_list, groups_list, proposals_list, text_type='group id', show_single_person=True):
    # 创建保存路径
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    filenames = []
    for root, dirs, files in os.walk(img_load_path):
        for file in files:
            if file[-3:] == 'jpg':
                filenames.append(file)
    filenames.sort()

    images = []
    print('Start to load all images!')
    # 把视频标注需要用到的图片载入内存，方便在上面绘制框
    for img_name in filenames:
        img_path = img_load_path + img_name
        # load img and resize
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        scale_ratio = IMAGE_VIS_WIDTH / width
        img_resized = cv2.resize(img, (int(width * scale_ratio), int(height * scale_ratio)))
        images.append(img_resized)
        print('Loaded image: ' + img_name.split('_')[-1])
    print('Images load finished!')

    # draw for each proposal
    for proposal_dict in proposals_list:
        proposal_id = proposal_dict['group_id']
        proposal_members_list = proposal_dict['person']

        # transfer color space from RGB to HSV to improve visual perception
        h = 30 * int(proposal_id) % 360
        s = 0.8 + random.random() * 0.2
        v = 0.35 + random.random() * 0.5
        r, g, b = hsv2rgb(h, s, v)

        for member_dict in proposal_members_list:
            id = member_dict['idx']
            for person_dict in persons_list:
                if person_dict['idx'] == id:
                    frames = person_dict['frames list']
                    for frame in frames:
                        frame_id = frame['frame_id']
                        x1 = int(frame['rect']['tl']['x'] * width * scale_ratio)
                        y1 = int(frame['rect']['tl']['y'] * height * scale_ratio)
                        x2 = int(frame['rect']['br']['x'] * width * scale_ratio)
                        y2 = int(frame['rect']['br']['y'] * height * scale_ratio)
                        cv2.rectangle(images[frame_id], (x1, y1), (x2, y2), (b, g, r), 1)
                        # show text
                        if text_type == 'person id':
                            cv2.putText(images[frame_id], str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (b, g, r), 1)
                        elif text_type == 'group id':
                            cv2.putText(images[frame_id], str(proposal_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (b, g, r), 1)

    # draw for each group
    for group_dict in groups_list:
        group_id = group_dict['group_id']
        group_type = group_dict['group_type']
        group_members_list = group_dict['person']

        # transfer color space from RGB to HSV to improve visual perception
        h = 30 * int(group_id) % 360
        s = 0.8 + random.random() * 0.2
        v = 0.35 + random.random() * 0.5
        r, g, b = hsv2rgb(h, s, v)

        if group_type != "single person":
            for member_dict in group_members_list:
                id = member_dict['idx']
                for person_dict in persons_list:
                    if person_dict['idx'] == id:
                        frames = person_dict['frames list']
                        for frame in frames:
                            frame_id = frame['frame_id']
                            x1 = int(frame['rect']['tl']['x'] * width * scale_ratio)
                            y1 = int(frame['rect']['tl']['y'] * height * scale_ratio)
                            x2 = int(frame['rect']['br']['x'] * width * scale_ratio)
                            y2 = int(frame['rect']['br']['y'] * height * scale_ratio)
                            # cv2.rectangle(images[frame_id], (x1, y1), (x2, y2), (b, g, r), 1)
                            # show text
                            if text_type == 'person id':
                                cv2.putText(images[frame_id], str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                            (b, g, r), 1)
                            elif text_type == 'group id':
                                cv2.putText(images[frame_id], str(group_id), (x1, y1 + 15),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                            (b, g, r), 1)

    print('Start to save painted images!')
    # 保存绘制结果图片
    for i, img in enumerate(images):
        cv2.imwrite(img_save_path + str(i).zfill(4) + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        print('Saved image: ' + str(i) + '.jpg')


if __name__ == '__main__':
    for i, person_gt_path in enumerate(PERSON_GT_JSON_PATHS_30):
        # skip HIT, because the frame rate of annotation of HIT is different (FPS = 6)
        if i != 0:
            continue

        img_urls = img_urls_get(GT_JSON_ROOT_PATH_30 + person_gt_path)
        persons_list = person_analysis(GT_JSON_ROOT_PATH_30 + person_gt_path)

        group_gt_path = GROUP_GT_JSON_PATHS_30[i]
        groups_list = group_analysis(GT_JSON_ROOT_PATH_30 + group_gt_path)

        # group_proposal_path = 'JSONs/Xili_Cross_group_proposal.json'
        # proposals_list = group_analysis(group_proposal_path)

        img_load_path = IMG_LOAD_ROOT_PATH_30 + SCENE_NAMES[i]
        img_save_path = IMG_SAVE_ROOT_PATH + img_load_path.split('/')[-2] + '/'
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)

        print('Start to paint on images from scene: ', SCENE_NAMES[i])

        # 若需要可视化的只有行人帧属性标注,使用tracking_draw; 若需要可视化帧属性和关系群组,使用group_draw
        # proposal_draw(img_save_path, img_load_path, persons_list, groups_list, proposals_list)
        # tracking_draw(img_save_path, img_load_path, persons_list, image_num=60)
        group_draw(img_save_path, img_load_path, persons_list, groups_list, image_num=20)
