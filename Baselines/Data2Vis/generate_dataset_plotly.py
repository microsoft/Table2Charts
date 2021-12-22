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
        print('File not exist:', file)
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
                print('Other Type:', df['fields'][i]['type'])
                value_type = 's'
            # value_type = data['records'][0][i][0]

            if value_type == 'n':
                if not name in head_num.keys():
                    head_num[name] = 'num' + str(len(head_num))
                new_name = head_num[name]
            elif value_type == 's' or 'str':
                if not name in head_str.keys():
                    head_str[name] = 'str' + str(len(head_str))
                new_name = head_str[name]
            else:
                # print(value_type)
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
                    # print(value)
                    value = round(value, 3)
                    # print(value)
                except ValueError:
                    pass
                li.append((name, value))  # format: [("num0","3.999"),("str0","Tim")]
            records.append(li)
        return records

    def get_chart_data(self, data, chart_type, type_first):
        if type_first:
            dic = dict()
            dic['type']=chart_type
            try:
                dic['y'] = [self.head_map[y['name']] for y in data['values']]
            except:
                return None
            try:
                dic['x'] = [self.head_map[x['name']] for x in data['categories']]
            except:
                return None
            if 'grouping' in data.keys():
                dic['group'] = self.map_group(data['grouping'])
        else:
            dic = dict()
            dic['type']=chart_type
            try:
                dic['y'] = [self.head_map[y['name']] for y in data['values']]
            except:
                return None
            try:
                dic['x'] = [self.head_map[x['name']] for x in data['categories']]
            except:
                return None
            if 'grouping' in data.keys():
                dic['group'] = self.map_group(data['grouping'])
        return dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='original plotly data')
    parser.add_argument('--save', type=str, default='', help='save dataset dir')
    parser.add_argument('--type_first', action='store_true', help='type at first, elas at last'
    args = parser.parse_args()

    dataset_path = args.data
    save_dir = args.save
    type_first=args.type_first

    lost_table = 0
    lost_chart = 0
    chart_error = 0

    li = load_json(dataset_path + 'test.json')
    print(len(li))
    print(li[:10])

    for stid in tqdm(li):
        settings = load_json(dataset_path + 'sample-new/' + stid.replace('.t0', '') + '.sample.json')
        if settings is None:
            lost_table += 1
            continue
        if 'tableAnalysisPairs' in settings.keys():
            pairs = settings['tableAnalysisPairs']
        elif 'tableAnalysesPairs' in settings.keys():
            pairs = settings['tableAnalysesPairs']
        else:
            print('lost')
            continue
        table_file = '{}.table.json'.format(stid)
        df_file = '{}.DF.json'.format(stid)

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

        tid = int(stid.split('.')[-1].strip('t'))
        chart_data_li = []
        for i in range(len(pairs[str(tid)])):
            cid = pairs[str(tid)][i]['index']
            chart_file = '{}.c{}.json'.format(stid, cid)
            chart_data = load_json(dataset_path + 'data/' + chart_file)
            if chart_data is None:
                lost_chart += 1
                continue
            chart_type = pairs[str(tid)][i]['anaType']
            chart_data = pair_data.get_chart_data(chart_data,chart_type,type_first,type_first)
            if chart_data is None:
                chart_error += 1
                continue
            chart_data_li.append(chart_data)
        if len(chart_data_li) == 0:  # if no charts in this table, abandon this table
            continue
        with open(save_dir + 'test_src.txt', 'a')as f:
            f.write(json.dumps(table_data_line))
            f.write('\n')
        with open(save_dir + 'test_tgt.txt', 'a')as f:
            f.write(json.dumps(chart_data_li))
            f.write('\n')
    print(lost_table, lost_chart, chart_error)
