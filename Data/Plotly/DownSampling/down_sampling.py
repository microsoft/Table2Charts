# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import collections
import multiprocessing
import os
import time
from collections import defaultdict
from down_sampler import proc_sample, eval_sampling_perf, sample_plotly
from tqdm import tqdm
from utils import transform_chart_type, dump_json, CORPUS, load_json, keep_plotly_table


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', type=str, required=True, help="The directory storing the dataset.")
    parser.add_argument('-l', '--language', type=str, required=True,
                        help="Choose the specific language, 'all' represent all languages.")
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=CORPUS,
                        help="Which dataset to be down sampled. Support 'plotly'.")
    parser.add_argument('-n', '--n_cores', type=int, default=1, help="Number of cores to process down sampling")
    parser.add_argument('--sample_name', type=str, default='sample-new', help="The name of sample file.")
    return parser.parse_args()


def main_plotly(args):
    src_path = args.source_path
    table_path = os.path.join(src_path, "table")
    chart_path = os.path.join(src_path, "data")
    output_path = os.path.join(src_path, args.sample_name)
    # schema : {"schema_id" : ["schema_id.t*.c*(p*).json"]}
    schema = {'.'.join(id.split('.')[:-3]): [] for id in
              os.listdir(table_path)}  # id: "schema_id.t*.table.json", but there may be "." in schema_id.
    for chart in os.listdir(chart_path):
        # find all "{sUid}.t{tUid}.c{cUid}.json" files.
        if chart.split('.')[-2][0] == 'c' and chart.split('.')[-3][0] == 't':
            schema['.'.join(chart.split('.')[:-3])].append(chart)
    charts_type = []
    chart_test_type = []
    test_list = list()
    schema_ids = list()
    prefix_type = defaultdict(int)
    if args.n_cores == 1:
        for key in tqdm(schema.keys()):
            chart_type, schema_id, prime_type = sample_plotly(src_path, key, schema[key], output_path)
            if chart_type is not None:
                charts_type.extend(chart_type)
                prefix = '_'.join(schema_id.split('_')[:-1])  # schema_id: "user_num"
                prefix_type[prefix + prime_type] += 1
                if keep_plotly_table(prefix, prime_type, prefix_type):
                    test_list.append(schema_id + '.t0')
                    schema_ids.append(schema_id)
                    chart_test_type.extend(chart_type)

    else:
        # multi-process
        src_list = [src_path] * len(schema)
        output_list = [output_path] * len(schema)
        pool = multiprocessing.Pool(args.n_cores)
        res_list = pool.starmap(sample_plotly, zip(src_list, schema.keys(), schema.values(), output_list))
        pool.close()
        pool.join()
        for chart_type, schema_id, prime_type in res_list:
            if chart_type is not None:
                charts_type.extend(chart_type)
                prefix = '_'.join(schema_id.split('_')[:-1])
                prefix_type[prefix + prime_type] += 1
                if keep_plotly_table(prefix, prime_type, prefix_type):
                    test_list.append(schema_id + '.t0')
                    schema_ids.append(schema_id)
                    chart_test_type.extend(chart_type)
    dump_json(test_list, os.path.join(src_path, 'test.json'))
    dump_json(schema_ids, os.path.join(src_path, 'index', 'schema_ids.json'))
    print(f"Original table number: {len(schema)}, chart number: {len(charts_type)}")
    print(f"Original chart type: {collections.Counter(charts_type)}")
    print(f"Test table number: {len(test_list)}, chart number: {len(chart_test_type)}")
    print(f"Test chart type: {collections.Counter(chart_test_type)}")


if __name__ == "__main__":
    args = get_arguments()
    if args.objective == 'plotly':
        main_plotly(args)
