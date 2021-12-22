# Table2Charts: Recommending Charts by Learning Shared Table Representations
This repository provides code and data of the paper _Table2Charts: Recommending Charts by Learning Shared Table Representations_.

## Table2Charts Code
The core parts included in the folder [`Table2Charts`](Table2Charts). See [`Table2Charts/README.md`](Table2Charts/README.md) for details.

## Baselines Code
In the paper Table2Charts is compared with the following four baselines:
* DeepEye: From the paper _DeepEye: Towards Automatic Data Visualization_ with inference models at <https://github.com/Thanksyy/DeepEye-APIs>.
* Data2Vis: From the paper _Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks_ with code at <https://github.com/victordibia/data2vis>.
* VizML: From the paper _VizML: A Machine Learning Approach to Visualization Recommendation_ with code and data at <https://github.com/mitmedialab/vizml>.
* DracoLearn: From the paper _Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco_ with inference models at <https://github.com/uwdata/draco>.

In the folder [`Baselines`](Baselines), we provide more details on how we train and evaluate those baselines.

## Data
In addition to our Excel chart corpus (which is under privacy reviews for publication), we use two more datasets for comparing with baselines in section 4.2:
* A public Plotly corpus used in VizML paper.
* 500 HTML tables (crawled from the public web) for human evaluation.

In the folder [`Data`](Data), we provide the way we get and process Plotly corpus, and the results about human evaluation.

## Results
We provide model in section 4.2.2 and human evaluation results in section 4.2.3.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
