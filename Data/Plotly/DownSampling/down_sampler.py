import collections
import os
from typing import List

from utils import load_json, transform_chart_type, dump_json


def sample_plotly(src_path, schema_id, chart_list, output_path):
    '''
    Generate schema_id.sample.json for schema.
    :param src_path: source directory path, where schema, chart, table information json file are stored.
    :param schema_id: Id of schema.
    :param chart_list: The list of schema_id's chart file name.
    :param output_path: output path, where schema_id.sample.json will be stored.
    '''
    ori_table_info = load_json(os.path.join(src_path, "table", schema_id + '.t0.table.json'))
    table_info = load_json(os.path.join(src_path, 'data', schema_id + '.t0.DF.json'))
    sampled_table_schema = {"sID": schema_id, "lang": 'en', "nColumns": ori_table_info['nColumns'],
                            "tableAnalysisPairs": {'0': []}}
    chart_type = []
    for chart in chart_list:
        chart_info = load_json(os.path.join(src_path, 'data', chart))
        for value in chart_info["values"]:
            # filter the charts whose value only have index=0 without "name", or whose value is string.
            if "name" not in value:
                break
            if table_info["fields"][value["index"]]["type"] == 1:
                break
        else:
            anaType = transform_chart_type(chart_info['cType'] + 'Chart', chart_info, table_info, ori_table_info)
            if not anaType or anaType == "scatterChart_notstring" or anaType == "lineChart_noline":
                continue
            else:
                chart_type.append(anaType)
            sampled_table_schema["tableAnalysisPairs"]["0"].append(
                {"anaType": anaType, "nVals": len(chart_info['values']), "index": chart.split('.')[-2][1:]})
    if len(sampled_table_schema["tableAnalysisPairs"]["0"]) != 0:
        dump_json(sampled_table_schema, os.path.join(output_path, schema_id + '.sample.json'))
        return chart_type, schema_id, max(collections.Counter(chart_type))
    else:
        return None, None, None
