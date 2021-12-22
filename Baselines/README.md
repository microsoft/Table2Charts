# Baselines
Table2Charts is compared with _DeepEye_, _Data2Vis_, _VizML_ and _DracoLearn_. Note that the scripts mentioned below are from their respective repositories unless otherwise noted.

## DeepEye
DeepEye only provides inference APIs. We clone [DeepEye repository](https://github.com/Thanksyy/DeepEye-APIs), and use its [`test.py`](https://github.com/Thanksyy/DeepEye-APIs/blob/master/test.py) to evaluate each table in our corpora.

## Data2Vis
We re-train and evaluate _Data2Vis_ with [Data2Vis repository](https://github.com/victordibia/data2vis). See details in [`Baselines/Data2Vis`](Data2Vis) directory in this repository.

## VizML
We re-train and evaluate _VizML_ with their code at [VizML repository](https://github.com/mitmedialab/vizml).
+ Extract features with [`feature_extraction/extract.py`](https://github.com/mitmedialab/vizml/blob/master/feature_extraction/extract.py).
+ Preprocess features with [`preprocessing/preprocess.py`](https://github.com/mitmedialab/vizml/blob/master/preprocessing/preprocess.py).
+ With the extracted features, we train and evaluate task 8 and task 11 (encoding level _Mark Type_ task and _Is on X-axis or Y-axis_ task) with [`neural_network/paper_tasks.py`](https://github.com/mitmedialab/vizml/blob/master/neural_network/paper_tasks.py). Note that we change its _Mark Type_ task from 3 classification to 4 classification (including line, scatter, bar, pie chart).

## Draco
We clone [Draco repository](https://github.com/uwdata/draco) and evaluate _Draco_ with [`Baselines/test_draco.py`](test_draco.py) in this repository (which is similar to [`tests/test_run.py`](https://github.com/uwdata/draco/blob/master/tests/test_run.py) in Draco repository). In [`Baselines/test_draco.py`](test_draco.py), you can know how we constrain the Draco model to generate charts.
