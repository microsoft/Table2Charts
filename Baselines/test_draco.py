# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import pytest
from draco.helper import read_data_to_asp
from draco.js import cql2asp
from draco.run import run
from jsonschema import validate
from tqdm import tqdm

EXAMPLES_DIR = os.path.join("examples")
PATH_INPUT = "/storage/chart-202011/chart-202011"
PATH_OUPUT = '/storage/draco-202101/excel1'
if not os.path.exists(PATH_OUPUT):
    os.mkdir(PATH_OUPUT)
with open(os.path.join(PATH_INPUT, "test.txt"), 'r', encoding="utf-8-sig") as f:
    files = json.load(f)


class TestFull:
    @pytest.mark.parametrize("file", files)
    def test_output_schema(self, file):
        query_spec = {
            "data": {
                "url": file + ".csv"
            },
            "mark": "?",
            "encodings": [
                {
                    "channel": "x",
                    "field": "?"
                },
                {
                    "channel": "y",
                    "field": "?"
                }
            ]
        }
        data = read_data_to_asp(os.path.join(PATH_INPUT, "csv", query_spec["data"]["url"]))
        query = cql2asp(query_spec)
        program = query + data
        result = run(program)

        try:
            with open(os.path.join(PATH_OUPUT, file + ".json"), 'w') as f:
                json.dump(result.as_vl(), f)
            print("success")
        except:
            print(f"!!!!!Error predict: {file}!!!!!")
