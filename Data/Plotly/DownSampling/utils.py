# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import random
from typing import Dict

CORPUS = ['plotly']

# Direct mapping between original excel chart types and transformed types
DIRECT_CTYPE_MAP = {
    'barChart': 'barChart',
    'bar3DChart': 'barChart',
    'stockChart': 'stockChart',
    'pieChart': 'pieChart',
    'pie3DChart': 'pieChart',
    'doughnutChart': 'pieChart',
    'ofPieChart': 'pieChart',
    'radarChart': 'radarChart',
    'bubbleChart': 'bubbleChart',
    'areaChart': 'areaChart',
    'area3DChart': 'areaChart',
    'surfaceChart': 'surfaceChart',
    'surface3DChart': 'surfaceChart'
}


def check_monotony(records):
    if len(records) < 2:
        return 0
    inc = (records[1] - records[0]) >= 0
    dec = (records[1] - records[0]) <= 0
    if inc:
        for i in range(1, len(records) - 1):
            if records[i + 1] - records[i] < 0:
                return 0
        return 1
    if dec:
        for i in range(1, len(records) - 1):
            if records[i + 1] - records[i] > 0:
                return 0
        return 2
    return 0


def transform_raw_line_chart(chart_info: Dict, table_info: Dict, ori_table_info: Dict):
    """
    For line chart and line-3D chart, the type will be either remained as line chart,
    or transform into series chart according to the type of category ("x")
    A line chart is actually a line chart if:
        (1) the values are connected by lines.
    """
    if 'valueDrawsLine' in chart_info and len(chart_info['valueDrawsLine']) > 0 and chart_info['valueDrawsLine'][0]:
        return "lineChart"
    else:
        return "lineChart_noline"


def transform_raw_scatter_chart(chart_info: Dict, table_info: Dict, ori_table_info: Dict):
    """
    For scatter chart, the type will be transformed into series chart, or remained as scatter
    according to the draw-line information
    Scatter is actually a line chart if:
        (1) With line connecting data points;
        (2.1) len(categories) != 1 or
        (2.2) the category is string/unknown or
        (2.3) the category field values are monotonic.
    Scatter is actually a scatter chart if:
        (1) Only 1 category;
        (2.1) With line connecting data points and the category field values are NOT monotonic, or
        (2.2) Without line connecting data points and the category is NOT string.
    Scatter is not sure if:
        (1) Only 1 category;
        (2) Without line connecting data points and the category is string.
    """
    if ('xFields' in chart_info) and len(chart_info['xFields']) == 1:
        cat_field_idx = chart_info['xFields'][0]['index']
        cat_type = table_info['fields'][cat_field_idx]['type']
        cat_records = get_cat_records(ori_table_info, cat_field_idx)
        single_cat = True
    else:
        return "lineChart"

    if 'valueDrawsLine' in chart_info and len(chart_info['valueDrawsLine']) > 0 and chart_info['valueDrawsLine'][0]:
        if single_cat:
            if cat_type in {0, 1}:
                return "lineChart"
            elif cat_type in {3, 5, 7}:
                try:
                    if check_monotony(cat_records) > 0:
                        return "lineChart"
                    else:
                        return "scatterChart"
                except:
                    return "lineChart"
        else:
            return "lineChart"
    else:
        if single_cat and cat_type in {0, 1}:
            return "scatterChart_notstring"
        else:
            return "scatterChart"


def transform_chart_type(ori_cType: str, chart_info: Dict, table_info: Dict, ori_table_info: Dict):
    """
    Transform the original chart types defined by Excel into the chart types used in papers.
    :param ori_cType: original chart type, defined by Excel.
    :param chart_info: chart information dictionary (from sID.tID.cID.json)
    :param table_info: table information dictionary (form sID.tID.DF.json)
    :param ori_table_info: origin table information dictionary (from sID.tID.table.json)
    :return: the transformed chart types in TARGET_CTYPE.
    """
    if ori_cType in DIRECT_CTYPE_MAP:
        return DIRECT_CTYPE_MAP[ori_cType]
    elif ori_cType in {'lineChart', 'line3DChart'}:
        return transform_raw_line_chart(chart_info, table_info, ori_table_info)
    elif ori_cType == 'scatterChart':
        return transform_raw_scatter_chart(chart_info, table_info, ori_table_info)
    else:
        return None


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        content = json.load(f)
    if content is None:
        raise Exception(f"Empty json: {file_name}")
    return content


def dump_json(obj, file_name, default=None):
    with open(file_name, 'w', encoding='utf-8-sig') as fw:
        if default is None:
            json.dump(obj, fw)
        else:
            json.dump(obj, fw, default=default)


def keep_plotly_table(prefix, prime_type, prefix_type):
    '''
    Whether to keep the (table,chart) pair in plotly dataset.
    When prime chart type is scatter chart, we keep 1 table for per 100 tables who have the same prefix + prime type with a probability of 0.5.
    When prime chart type is line chart, we keep 1 table for per 10 tables who who have the same prefix + prime type.
    When prime chart type is bar or pie chart, we keep all table.
    :param prefix: prefix of plotly suid.(suid: "user_num", prefix: "user")
    :param prime_type: prime chart type of the table.
    :param prefix_type: number of tables who have the same prefix + prime type
    '''
    return (prime_type == "scatterChart" and prefix_type[
        prefix + prime_type] % 100 == 1 and random.random() > 0.5) or (
                   prime_type == "lineChart" and prefix_type[prefix + prime_type] % 10 == 1) or (
                   prime_type == "barChart" and prefix_type[prefix + prime_type] % 1 == 0) or (
                   prime_type == "pieChart" and prefix_type[prefix + prime_type] % 1 == 0)
