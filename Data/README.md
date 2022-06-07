# Data
In addition to our Excel chart corpus, we use public Plotly corpus to evaluate baselines and HTML tables (crawled from the public web) to do human evaluation. 


## Plotly corpus
The public Plotly corpus is available at [VizML repository](https://github.com/mitmedialab/vizml). Following steps are the details on preparing Plotly data:
+ Download its full Plotly corpus with [`retrieve_data.sh`](https://github.com/mitmedialab/vizml/blob/master/retrieve_data.sh) in VizML repository.
+ Deduplicate data with its [`data_cleaning`](https://github.com/mitmedialab/vizml/tree/master/data_cleaning) directory. 
+ The same procedures of combo chart splitting and down sampling are applied to the remaining (table, charts) pairs as in Excel corpus.
    + Chart splitting: Begin chart splitting with [`Data/Plotly/ChartSplit/HandlePlotlyTable.cs`](Plotly/ChartSplit/HandlePlotlyTable.cs) in this repository. `ExtractForPlotlyTablesAll()` is the entrance of splitting all (table, chart) pairs from a TSV file, and `ExtractForPlotlyTables()` is the entrance of splitting one (table, chart) pair from JSON file.
    + Down sampling: See details in [`Data/Plotly/DownSampling`](Plotly/DownSampling) folder in this repository.


## Human Evaluation Data
We crawl 500 public web HTML tables with different schema, and after human evaluation we get 330 tables who are suitable for generating charts. In the JSON file [`Data/HumanEvaluation/human_eval_results.json`](Results/HumanEvaluation/human_eval_results.json), there are the original 330 tables, their corresponding charts recommended by Table2Charts, Data2Vis & DeepEye, and human evaluation results.

The result of one (table, charts) being evaluated is organized as the following format:
```JSON
{
    "Table": {
        "Url": "...",
        "Header": ["..."],
        "Value": [["..."],["..."]]
    },
    "Table2Chart chart": {
        "ANA": "...",
        "Y": ["...", "..."],
        "X": ["..."],
        "GRP": ["..."],
        "score": 0.8
    },
    "DeepEye chart": {
        ...
    },
    "Data2Vis chart": {
        ...
    },
    "Table2Chart ratings": [5, 5, 5],
    "DeepEye ratings": [4, 4, 4],
    "Data2Vis ratings": [3, 3, 3]
}
```
For a table, `Url`, `Header`, `Values` mean the webpage url the table comes from, the header of the table, and the values of the table.
For a chart, `ANA`, `X`, `Y`, `GRP`, `score` mean the chart type, x fields, y fields, grouping type, and confidence score of recommending this chart. 
The ratings for a chart from all three raters are stored in a list.


[`Results/HumanEvaluation/humanEvaluation.py`](Results/HumanEvaluation/humanEvaluation.py) gives the distribution of the ratings of the three systems, conducts Wilcoxon signed-rank test and computes Cliff's delta effect size for comparison between Table2Charts and the two other systems.

The tables and human evaluation labels are published under the [CDLA license](CDLA-Permissive-2.0.md).
