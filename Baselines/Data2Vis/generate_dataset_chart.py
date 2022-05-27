# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
import random

from tqdm import tqdm

GROUP_MAP = {
    "standard": "cluster",
    "clustered": "cluster",
    "stacked": "stack",
    "percentStacked": "stack"
}


def load_json(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    if data.startswith(u'\ufeff'):
        data = data.encode('utf8')[3:].decode('utf8')
    data = json.loads(data)
    return data


class PairData():
    def __init__(self):
        self.head_map = {}
        self.head_idx = {}

    def map_group(self, group):
        if group not in GROUP_MAP.keys():
            return None
        else:
            return GROUP_MAP[group]

    def construct_map(self, df):
        # normalize head
        head_num = {}
        head_str = {}
        for i in range(len(df['fields'])):
            name = df['fields'][i]['name']  # old head
            if df['fields'][i]['type'] in [0, 1]:
                value_type = 's'
            elif df['fields'][i]['type'] in [3, 5, 7]:
                value_type = 'n'
            else:
                print(df['fields'][i]['type'])
                value_type = 's'

            if value_type == 'n':
                if name not in head_num.keys():
                    head_num[name] = 'num' + str(len(head_num))
                new_name = head_num[name]
            elif value_type == 's' or 'str':
                if name not in head_str.keys():
                    head_str[name] = 'str' + str(len(head_str))
                new_name = head_str[name]
            else:
                print(value_type)
                new_name = df['fields'][i]['type']
            self.head_map[name] = new_name
            self.head_idx[new_name] = i

    def get_table_data(self, data):
        # all records
        records = []
        for i in range(len(data['records'])):
            li = []
            for j in range(len(data['fields'])):
                name = self.head_map[data['fields'][j]['name']]
                value = data['records'][i][j][1]
                try:
                    value = float(value)
                    value = round(value, 3)
                except ValueError:
                    pass
                li.append((name, value))  # format: [("num0","3.999"),("str0","Tim")]
            records.append(li)
        return records

    def get_chart_data(self, data):
        dic = dict()
        dic['y'] = [self.head_map[y['name']] for y in data['yFields']]
        dic['x'] = [self.head_map[y['name']] for y in data['xFields']]
        if 'grouping' in data.keys():
            dic['group'] = self.map_group(data['grouping'])
        return dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='original_data/', help='original excel data')
    parser.add_argument('--save', type=str, default='dataset/', help='save dataset dir')
    parser.add_argument('--n_sample', type=int, default=1)
    args = parser.parse_args()

    dataset_path = args.data
    save_dir = args.save
    sample_n = args.n_sample

    lost_table = 0
    lost_chart = 0

    # make dir 
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # schema ids
    sids = []
    for file in os.listdir(dataset_path + 'sample-new/'):
        sids.append(file.split('.')[0])

    test_tids = load_json(dataset_path + 'test.txt')

    for stid in tqdm(test_tids):
        sid = int(stid.split('.')[0])
        tid = int(stid.split('.')[1].strip('t'))

        settings = load_json(dataset_path + 'sample-new/' + str(sid) + '.sample.json')

        table_file = '{}.t{}.table.json'.format(sid, tid)
        df_file = '{}.t{}.DF.json'.format(sid, tid)
        table_data = load_json(dataset_path + 'table/' + table_file)
        table_df = load_json(dataset_path + 'data/' + df_file)
        if table_data is None:
            lost_table += 1
            continue
        pair_data = PairData()
        pair_data.construct_map(table_df)
        table_data = pair_data.get_table_data(table_data)

        idx = random.randint(0, len(table_data) - 1)
        table_data_line = table_data[idx]

        with open(save_dir + '/test_src.txt', 'a')as f:
            f.write(json.dumps(table_data_line))
            f.write('\n')

        with open(save_dir + "/head_idx.txt", "a")as f:
            f.write(json.dumps(pair_data.head_idx))
            f.write("\n")

    print("lost tables:", lost_table, "lost charts:", lost_chart)
