# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json

from tqdm import tqdm


def cmp_li(li1, li2):
    if len(li1) == 0 and len(li2) > 0:
        return False
    if len(li1) > 0 and len(li2) == 0:
        return False
    if len(li1) == 0 and len(li2) == 0:
        return True
    if li1[0] == li2[0]:
        return cmp_li(li1[1:], li2[1:])
    else:
        return False


def equal(pred, tgt):
    if not isinstance(pred, dict):
        return False
    if not ('type' in pred.keys() and 'y' in pred.keys() and 'x' in pred.keys()):
        return False
    if 'group' in tgt.keys() and 'group' not in pred.keys():
        return False
    if 'group' in pred.keys() and 'group' not in tgt.keys():
        return False
    # check x and y
    if not isinstance(pred['y'], list):
        return False
    if not isinstance(pred['x'], list):
        return False
    if len(pred['y']) != len(tgt['y']) or len(pred['x']) != len(tgt['x']):
        return False
    if set(pred['y']) != set(tgt['y']):  # y has no order
        return False
    if not cmp_li(pred['x'], tgt['x']):  # compare list elementwise, x has order
        return False
    # check type
    if not pred['type'] == tgt['type']:
        return False
    # check group
    if 'group' in tgt.keys() and pred['group'] != tgt['group']:
        return False
    return True


def field_selection_equal(pred, tgt):
    if not isinstance(pred, dict):
        return False
    # pred should have y x keys
    if not ('y' in pred.keys() and 'x' in pred.keys()):
        return False
    if not isinstance(pred['y'], list):
        return False
    if not isinstance(pred['x'], list):
        return False
    # check n fields
    if len(pred['y']) + len(pred['x']) != len(tgt['y']) + len(tgt['x']):
        return False
    pred_fields = set(pred['y']) | set(pred['x'])
    tgt_fields = set(tgt['y']) | set(tgt['x'])
    if pred_fields != tgt_fields:
        return False
    return True


def vis_encoding_equal(pred, tgt):
    if not ('type' in pred.keys() and 'y' in pred.keys() and 'x' in pred.keys()):
        return False
    if 'group' in tgt.keys() and 'group' not in pred.keys():
        return False
    if 'group' in pred.keys() and 'group' not in tgt.keys():
        return False
    # field mapping
    if len(pred['y']) != len(tgt['y']) or len(pred['x']) != len(tgt['x']):
        return False
    if set(pred['y']) != set(tgt['y']):
        return False
    if not cmp_li(pred['x'], tgt['x']):  # compare list elementwise
        return False
    # type and grouping
    if not pred['type'] == tgt['type']:
        return False
    if 'group' in tgt.keys() and pred['group'] != tgt['group']:
        return False
    return True


def to_json(s):
    try:
        chart = json.loads(s)
    except:
        return None
    return chart


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str,
                        default='pred.txt', help='predction result file')
    parser.add_argument('--target', type=str, default='test_tgt.txt', help='target file')
    parser.add_argument('--debug', action="store_true", help='debug mode')
    parser.add_argument('--cnt_emp', action="store_true",
                        help='count the result without any valid chart recommendation in task2')
    args = parser.parse_args()

    # print result
    debug = args.debug
    k = 20
    k1 = 0
    k2 = 0
    k0 = 0
    task0_pos = []
    task1_pos = []
    task2_pos = []
    task0_neg = []
    task1_neg = []
    task2_neg = []

    # load all examples
    pred_file = args.pred
    tgt_file = args.target
    print(pred_file, tgt_file)

    all_tgt = []
    all_pred = []
    with open(tgt_file, 'r')as f:
        for line in f.readlines():
            line = line.strip("\n")
            all_tgt.append(line)
    with open(pred_file, 'r')as f:
        for line in f.readlines():
            line = line.strip("\n")
            all_pred.append(line)
    print(len(all_pred), len(all_tgt))

    n = 0
    n_c = 0  # n charts
    n_s = 0  # n of scatter chart
    n_all = 0

    valid = 0
    task0_recall = [0, 0, 0]
    task1_recall = [0, 0, 0]
    task2_recall = [0, 0, 0]
    avg_len = 0

    # one example
    for i in tqdm(range(len(all_pred))):
        tgt = all_tgt[i]
        pred = all_pred[i]  # pred is a list of strings
        try:
            tgt_li = json.loads(tgt)
        except:
            print('tgt:', tgt)
            continue
        try:
            pred_li = json.loads(pred)
        except:
            continue

        result0 = []
        result1 = []
        for p in pred_li:  # beamsearch size predictions
            n += 1
            avg_len += len(p)
            pred_chart = to_json(p)
            if pred_chart:  # valid json
                valid += 1

                # task 0
                flag0 = False
                for tgt_chart in tgt_li:
                    if equal(pred_chart, tgt_chart):
                        flag0 = True
                result0.append(int(flag0))

                # task 1
                flag1 = False
                for tgt_chart in tgt_li:
                    if field_selection_equal(pred_chart, tgt_chart):
                        flag1 = True
                result1.append(int(flag1))

            else:
                result0.append(0)
                result1.append(0)

        task0_recall[0] += int(bool(sum(result0[:1])))
        task0_recall[1] += int(bool(sum(result0[:3])))
        task0_recall[2] += int(bool(sum(result0[:5])))

        task1_recall[0] += int(bool(sum(result1[:1])))
        task1_recall[1] += int(bool(sum(result1[:3])))
        task1_recall[2] += int(bool(sum(result1[:5])))

        if debug:
            if int(bool(sum(result0[:1]))) == 1:
                n_all += 1
                if to_json(pred_li[0])["type"] == "scatterChart":
                    n_s += 1
            if int(bool(sum(result0[:1]))) == 1 and k0 < k:
                task0_pos.append((i, pred_li[0], tgt_li))
                k0 += 1
            if int(bool(sum(result1[:1]))) == 1 and k1 < k:
                task1_pos.append((i, pred_li[0], tgt_li))
                k1 += 1

        # task 2
        pattern_dic = {}
        for tgt_chart in tgt_li:
            pattern = tuple(sorted(list(set(tgt_chart["x"] + tgt_chart["y"]))))
            if pattern not in pattern_dic.keys():
                pattern_dic[pattern] = [tgt_chart]
            else:
                pattern_dic[pattern].append(tgt_chart)

        # for tgt_chart in tgt_li:
        for tgt_chart_li in pattern_dic.values():
            topk = []
            for p in pred_li:
                pred_chart = to_json(p)
                if not pred_chart:
                    continue
                if field_selection_equal(pred_chart, tgt_chart_li[0]):
                    topk.append(pred_chart)
            if len(topk) == 0:
                if args.cnt_emp:
                    n_c += 1
            else:
                n_c += 1
            result2 = []
            for pred_chart in topk:
                in_tgt = False
                for tgt_chart in tgt_chart_li:
                    if vis_encoding_equal(pred_chart, tgt_chart):
                        in_tgt = True
                if in_tgt:
                    result2.append(1)
                else:
                    result2.append(0)
            task2_recall[0] += int(bool(sum(result2[:1])))
            task2_recall[1] += int(bool(sum(result2[:3])))
            task2_recall[2] += int(bool(sum(result2[:5])))
            if debug:
                if int(bool(sum(result2[:1]))) == 1 and k2 < k:
                    task2_pos.append((i, topk[0], tgt_chart_li))
                    k2 += 1

    n_t = len(all_pred)  # n_tables
    print('n tables:', n_t)
    print('avg pred len:', avg_len / n)
    print('valid:', valid / n)
    print('task0 recall@1,3,5:', task0_recall[0] / n_t, task0_recall[1] / n_t, task0_recall[2] / n_t)
    print('task1 recall@1,3,5:', task1_recall[0] / n_t, task1_recall[1] / n_t, task1_recall[2] / n_t)
    print('task2 recall@1,3,5:', task2_recall[0] / n_c, task2_recall[1] / n_c, task2_recall[2] / n_c)
    if debug:
        print("% scatter chart in correct task0:", n_s / n_all)
        print("task0 pos")
        for item in task0_pos:
            print(item)
        print("task1 pos")
        for item in task1_pos:
            print(item)
        print("task2 pos")
        for item in task2_pos:
            print(item)
