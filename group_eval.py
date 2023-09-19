"""
Date: 2019/9/25
Author: wxy
Function: computes precision and recall scores, as well as the number of true positives, false positives and false
negatives, give a set of detected groups and a set of ground truth groups.
"""
import tqdm

# Input:
# GROUP = <G-elements> list, each element contains a group detected by tested algorithm, each one defined by the
#         array of subjects' ID of individuals belonging to it.
# GT = <G-elements> list, each element contains a group provided by ground truth, each one defined by the array of
#       subjects' ID of individuals belonging to it.
# crit = string, it defines the criterium to establish whether group has been detected or not. It can be:
#        - 'card' : the detected group have to contain at least 2/3 of the elements of GT group (and vice-versa).
#        - 'all'  : the detected group and GT group have to match perfectly.
#        - 'dpmm': at least the 60% of the member are correctly groupend (T=0.6);
#
# Output:
# P = the ability to detect 'only' the ground truth (or the ability not to generate false positives).
#      Pr = TP / (TP+FP)
# R = the ability to detect 'all' the ground truth (or the ability not to generate false negatives).
#      Re = TP / (TP+FN)
# F1 = 2PR / (P+R)
# TP = number of True Positives
# FP = number of False Positives
# FN = number of False Negatives
def group_eval(GROUP, GT, crit='card'):
    # check group and gt
    # if not GROUP:
    #     print('group is empty!')
    #     return
    if not GT:
        print('gt is empty!')
        return

    TP = 0
    # print("Evaluating Grouping Performance...")
    for gt in GT:
        gt_set = set(gt)
        gt_card = len(gt)
        for group in GROUP:
            group_set = set(group)
            group_card = len(group)
            if crit == 'half':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 1/2:
                    TP += 1
            elif crit == 'card':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 2/3:
                    TP += 1
            elif crit == 'dpmm':
                inters = list(gt_set & group_set)
                inters_card = len(inters)
                if group_card == 2 and gt_card == 2:
                    if not len(gt_set - group_set):
                        TP += 1
                elif inters_card / max(gt_card, group_card) > 0.6:
                    TP += 1
            elif crit == 'all':
                if not len(gt_set - group_set):
                    TP += 1

    FP = len(GROUP) - TP
    FN = len(GT) - TP

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1, TP, FP, FN
