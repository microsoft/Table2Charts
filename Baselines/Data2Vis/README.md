# Run Data2Vis
 
## Input and Output Normalization
The header names are replaced by a shortened string, both for input and output sequences. The shortened strings are generated for each table. An example is shown as bellow.

 ```shell
Input: [["str0", "Use of ICT"], ["num0", "1.27"], ["num1", "1.62"], ["num2", "2.33"], ["num3", "4"]]
Output: [{"type": "bar3D", "y": ["str0"], "x": ["num0", "num1", "num2"], "group": "clustered"}]
 ```

## Setup
 - Clone [Data2Vis](https://github.com/victordibia/data2vis) repository to this directory.
 - Replace with new config files.
 ```shell
 mv chart.yml data2vis/example_configs/
 mv chart_infer.yml data2vis/example_configs/
 ```
 - Save results of Data2Vis. Change `_save_prediction_to_dict()` function in `data2vis/bin/infer.py` as follows:
 ```python
def _save_prediction_to_dict(output_string):
    if not os.path.exists(FLAGS.model_dir+'/pred'):
        os.makedirs(FLAGS.model_dir+'/pred')
    with open(FLAGS.model_dir+'/pred/'+FLAGS.save_file,'a')as f:
        f.write(json.dumps(output_string))
        f.write('\n')
 ```
 - Generate dataset and vocabulary for training and inference.
 ```shell
python generate_dataset_plotly.py --data <dataset_path> --save <data_dir> --n_sample 2
python data2vis/generate_vocab.py <data_dir>/all_src.txt --max_vocab_size 100 --delimiter > vocab_src.txt''
python data2vis/generate_vocab.py <data_dir>/all_tgt.txt --max_vocab_size 100 --delimiter > vocab_tgt.txt ''
 ```
## Training
```shell
python3 -m bin.train --config_paths="example_configs/chart.yml,example_configs/train_seq2seq.yml,example_configs/text_metrics_bpe.yml" --output_dir=<model_dir>
```
## Inference
```shell
python3 -m bin.infer --config_path="example_configs/chart_infer.yml" --model_dir <model_dir> --save_file <save_file>
```
## Evaluation
```shell
python tasks_all.py --pred <pred_file> --target <data_dir>/test_tgt.txt --cnt_emp
```
Task 0 gives the overall performance, task 1 evaluates field selection, task 2 evaluates design choices.
