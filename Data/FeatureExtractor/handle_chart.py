# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from data_feature_extractor import DataFeatureExtractor


class HandleChart():
    def __init__(self):
        pass

    def ExtractForChart(self, data_path: str, output_path: str, uid: str):
        print("Extracting features for ChartTables in {}.".format(uid))
        # edge case: no charts found, delete all json files and exit.

        # 1. Read from the <uid>.json for table information (lang)
        with open(data_path + uid + '.json', 'r', encoding='utf-8-sig') as json_file:
            schema = json.loads(json_file.read())

        # Get the language of the schema
        if 'lang' not in schema or schema['lang'] == None:
            language = 'mul'
        elif schema['lang'].startswith('zh'):
            language = 'zh'
        else:
            language = schema['lang']

        # Get the unique table number of the schema 
        unique_table_number = len(schema['uniqueTables'])

        # 2. Run features for each table 
        for table_idx in range(unique_table_number):
            newTUid = '{}.t{}'.format(uid, str(table_idx))
            with open(data_path + newTUid + '.table.json', 'r', encoding='utf-8-sig') as json_file:
                ct = json.loads(json_file.read())
            source_features = DataFeatureExtractor.ExtractTableFeatures(ct, language)
            source_features.delete_dt()

            # 4. dump features 
            with open(output_path + newTUid + '.DF.json', 'w') as f:
                f.write(json.dumps(source_features.__dict__))

            if table_idx > 0:
                return
