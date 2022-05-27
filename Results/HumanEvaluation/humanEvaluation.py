# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wilcoxon

all_model = ['Table2Chart', 'DeepEye', 'Data2Vis']


def load_json(data):
    try:
        if data.startswith(u'\ufeff'):
            data = data.encode('utf8')[3:].decode('utf8')
        data = json.loads(data)
        return data
    except:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flt', action="store_true", help="remove tables unsuitable for making a chart.")
    parser.add_argument('--n', type=int, default=1,
                        help="tables with n users marking 'unsuitable for chart' will be filtered. Default=1.")
    parser.add_argument('--t_mean', action="store_true",
                        help="whether to use the mean score or all scores of 3 raters for one table for wilcoxon tests and cliff delta score.")
    parser.add_argument('--s_mean', action="store_true",
                        help="whether to use the mean score according to schemas for wilcoxon tests and cliff delta score.")
    parser.add_argument('--file', type=str, default='UHRS_Task_MultiModelLabel_3Overlap_20210126.tsv',
                        help='human evaluation result file')

    args = parser.parse_args()

    filter_table = args.flt
    unsuit_n = args.n
    table_mean = args.t_mean
    combine_schema = args.s_mean
    result_file = args.file

    print('filter tables:', filter_table, 'n=', unsuit_n)
    print('use table mean:', table_mean)
    print('use schema mean:', combine_schema)

    batch_samples = []
    results = {}  # n_plot * 3
    sids = {}
    result_info_ori = {}  # {url: {"Table":  ,"Table2Chart chart":  ,"DeepEye chart":  ,"Data2Vis chart":  ,"Table2Chart ratings": [5, 5, 5],"DeepEye ratings": [4, 4, 4],"Data2Vis ratings": [3, 3, 3]}

    tsv = pd.read_csv(result_file, sep='\t')
    batch_samples.append(len(tsv.index))
    for i in tsv.index:  # one person, one plot, 3 models
        Table = load_json(tsv['MultiResJson'][i])
        url = Table['Table']['Url']
        Reco = load_json(tsv['RecoJson'][i])
        model1 = Reco[0] if Reco[0] != "" else {"modelName": "DeepEye"}
        model2 = Reco[1]
        model3 = Reco[2]
        scores = load_json(tsv['AnswerJson'][i])
        # save table infomation
        if url not in result_info_ori:
            table = load_json(tsv['MultiResJson'][i])
            result_info_ori[url] = {}
            values_ori = table["Table"]["Values"]
            values = []
            for j in values_ori:
                value = []
                for k in j:
                    value.append(k["text"])
                values.append(value)
            table["Table"]["Values"] = values
            result_info_ori[url]["Table"] = table["Table"]
            del result_info_ori[url]["Table"]['tUid']
            result_info_ori[url]["Table2Chart chart"] = table["Table2Chart"]
            del result_info_ori[url]["Table2Chart chart"]["ModelName"]
            result_info_ori[url]["DeepEye chart"] = table["DeepEye"]
            del result_info_ori[url]["DeepEye chart"]["ModelName"]
            result_info_ori[url]["Data2Vis chart"] = table["Data2Vis"]
            del result_info_ori[url]["Data2Vis chart"]["ModelName"]
        k = 0
        if scores:  # if has result
            if url not in results.keys():
                results[url] = {'Table2Chart': [], 'DeepEye': [], 'Data2Vis': [], 'unsuit': 0, 'sid': None,
                                'hits': []}

            results[url]['hits'].append(int(tsv['HitID'][i]))

            # remove unsuitable
            if tsv['NotSuitableForChart'][i] == True:
                results[url]['unsuit'] += 1

            # sid
            sid = url
            results[url]['sid'] = sid
            if sid not in sids.keys():
                n_sid = len(sids)
                sids[sid] = n_sid
            # scores
            model_name = {'Table2Chart': 0, 'DeepEye': 0, 'Data2Vis': 0}
            null_id = -1
            for j in range(3):
                model = [model1, model2, model3][j]
                if model != None:
                    if model != "":
                        model = model['modelName']
                        model_name[model] = 1
                        results[url][model].append(scores[k])
                    else:
                        null_id = k
                    k += 1
            if null_id != -1:
                null_model_name = [k for k, v in model_name.items() if v == 0]
                if len(null_model_name) != 1:
                    raise ValueError("There are more than 1 model that don't have chart.")
                results[url][null_model_name[0]].append(scores[null_id])

    print('all samples:', sum(batch_samples))
    for i in batch_samples:
        print('batch samples:', i)

    # stat
    print('tables:', len(results))
    print('schemas:', len(sids))

    # remove unsuitable tables
    if filter_table:
        filtered_results = results.copy()
        for url, result in results.items():
            if result['unsuit'] >= unsuit_n:
                del filtered_results[url]
        results = filtered_results
        print('suitable tables:', len(results))
        print('\n')

    res = [[], [], []]
    result_info = []
    for table, scores in results.items():
        res[0].extend(scores['Table2Chart'])
        res[1].extend(scores['DeepEye'])
        res[2].extend(scores['Data2Vis'])
        info = result_info_ori[table]
        info["Table2Chart ratings"] = scores['Table2Chart']
        info["DeepEye ratings"] = scores['DeepEye']
        info["Data2Vis ratings"] = scores['Data2Vis']
        result_info.append(info)
    with open('human_eval_results.json', 'w', encoding='utf-8-sig') as f:
        json.dump(result_info, f)

    if combine_schema:
        res_tmp = [[[] for i in range(len(sids))] for j in range(3)]
        for url, result in results.items():
            sid = result['sid']
            idx = sids[sid]
            res_tmp[0][idx].extend(result['Table2Chart'])
            res_tmp[1][idx].extend(result['DeepEye'])
            res_tmp[2][idx].extend(result['Data2Vis'])
        res_tmp = np.array(res_tmp)

        filtered = []
        for i in range(len(res_tmp[0])):
            if not res_tmp[0][i] == []:
                filtered.append(i)
        res_tmp = res_tmp[:, filtered]
        print('# schema:', len(filtered))

        res_mean = []
        for model_res in res_tmp:
            li = []
            for schema_res in model_res:
                mean = np.mean(schema_res)
                li.append(mean)
            res_mean.append(li)
        res_mean = np.array(res_mean)
    else:
        res_mean = [[], [], []]
        for table, scores in results.items():
            res_mean[0].append(np.mean(scores['Table2Chart']))
            res_mean[1].append(np.mean(scores['DeepEye']))
            res_mean[2].append(np.mean(scores['Data2Vis']))

    # cnt
    for i in range(3):
        print(all_model[i])
        cnt = Counter(res[i])
        print(Counter(res[i]), end='\t')
        print('mean:', np.mean(res[i]), end='\t')
        print('median:', np.median(res[i]), end='\t')
        more_3 = 0
        more_4 = 0
        less_2 = 0
        for k, v in cnt.items():
            if k >= 3:
                more_3 += v
            if k >= 4:
                more_4 += v
            if k <= 2:
                less_2 += v
        print('>=3:', more_3, '\t>=4:', more_4, '\t<=2:', less_2)

    if table_mean:
        data1 = res_mean[0]
        data2 = res_mean[1]
        data3 = res_mean[2]
    else:
        data1 = res[0]
        data2 = res[1]
        data3 = res[2]
    print('\n')
    for i in range(2):
        d1, d2 = [(data1, data2), (data1, data3)][i]
        print(['Table2Chart vs DeepEye', 'Table2Chart vs Data2Vis'][i])
        for alt in ['two-sided', 'greater', 'less']:
            stat, p = wilcoxon(d1, d2, alternative=alt)  # alternative=alt
            print('wilcoxon ({}): p={}'.format(alt, p))
        print('\n')
