# Usage of Down Sampling

This folder include 3 python files:
- **down_sampling.py**: The entrance of down sampling process.
- **down_sampler.py**: Correlated methods.
- **utils.py**: Some utility methods.

Run the following code to conduct down sampling:

```bash
python down_sampling.py -s <source_path> -l <language constraints> -o <chart/pivot> -n <number_cores>
```

One need to determined the *source path* and the *target path*, *language constraints*, *objectives*

- In *source path*, it should contain the following JSON files:
  - `schema_id.json`: which include the schema information.
  - `schma_id.table_id.DF.json`: which include the information of  chart table.
  - `schema_id.table_id.chart_id.json`: which include the information of analysis chart

- After down sampling of each schema, there will be a corresponding `schema.sample.json`, which include the information after down sampling, exists in the *target path*.
- The *language constraints* indicates only perform down sampling over the schemas with the specific language. It can be either single type of language (like "en", etc.) or all languages (use "all" for this parameter).
- The *objective* refers to which kind of schema the down sampling is conducted over. It can only be Plotly in this task.
- One can also specify the *number_cores* to parallel conduct the down sampling. If not specified or specified as 1, then the down sampling will be conducted in a serialized manner.
