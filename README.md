# Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations

In [[1]](#1) we further develop the work of [[2]](#2) and [[3]](#3), which provide a set of evaluation metrics for saliency methods. We extend this set to a comprehensive list of metrics that mimic common metrics for classification evaluation based on the definition of correct and incorrect feature importance in images. Particularly, the following points are addressed:

- We include saliency metrics that produce interesting results (e.g. specificity), but were overlooked in [[3]](#3)
- In addition to the saliency methods discussed in [[2]](#2), we also include SHAP [[4]](#4) and B-cos [[5]](#5)
- We show how such metrics can be evaluated using Krippendorff's $\alpha$ [[6]](#6) and Spearman's $\rho$ [[7]](#7)[[8]](#8) instead of taking them at face value (which is already a problem in XAI, as discussed in the paper)

This repository contains the code and datasets that are needed to recreate the experiments conducted in our paper: [Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations](https://dl.acm.org/doi/10.1145/3630106.3658537). As our paper builds on [[2]](#2), this repository also builds on their implementation of the [Focus-metric](https://github.com/HPAI-BSC/Focus-Metric).

### Explainability methods: 

- KernelSHAP from [Captum](https://captum.ai/api/kernel_shap.html) of Kokhlikyan *et al.* [[15]](#15) as a representative of the SHAP-family.
- B-cos networks version 2 based on the [work](https://github.com/B-cos/B-cos-v2) of Boehle *et al.* [[5]](#5).
- Smoothgrad [[9]](#9); implementation based on this [repo](https://github.com/wjNam/Relative_Attributing_Propagation) of Nam *et al.* 
- Layer-wise Relevance Propagation (LRP) [[10]](#10); implementation based on this [repo](https://github.com/kazuto1011/grad-cam-pytorch) of Nakashima *et al.*
- GradCAM [[11]](#11); implementation based on this [repo](https://github.com/jacobgil/pytorch-grad-cam)
of Gildenblat *et al.*
- LIME [[12]](#12); implementation based on this [repo](https://github.com/marcotcr/lime) of Tulio *et al.*
- GradCAM++ [[13]](#13); implementation based on this [repo](https://github.com/jacobgil/pytorch-grad-cam) of Gildenblat *et al.*
- Integrated Gradients (IG) [[14]](#14); implementation based on [Captum](https://github.com/pytorch/captum) of Kokhlikyan *et al.* [[15]](#15).


The first two saliency methods (KernelSHAP and B-cos) were added by us, the other six saliency methods have been adopted unchanged from the [Focus-metric](https://github.com/HPAI-BSC/Focus-Metric) repository.


### Requirements

This code runs under Python 3.10.4. The python dependencies are defined in `requirements.txt`. 


## Available mosaics

We provide two data sets (located in folder `data`) that can be used to generate mosaics for the XAI evaluation (this is done by executing the script `xai_eval_scripy.py`; detailed instructions on how to run experiments see *How to run experiments*). The authors of [[2]](#2) provide mosaics in their [repository](https://github.com/HPAI-BSC/Focus-Metric#requirements), which can also be used with our code. To do so, download the mosaics and copy them to `data/mosaics/`. When executing `xai_eval_script.py` the **--dataset** argument has to correspond to the mosaics (i.e. **--dataset** ilsvrc2012 for **--mosaics** ilsvrc2012_mosaics) and the **--mosaics_per_class** argument has to be None.

### Dataset instructions

The models used here are all trained on ImageNet and are not fine-tuned. Therefore, new data sets should consist of classes that are available in ImageNet. To create a new data set so that it can be used with the code, perform the following steps:

1. 	Create subfolder in `data/datasets/` with the dataset name.

2. 	Add folder `data`, which contains images that will be used for mosaic creation. 
	Images should be named *ClassName-ImageNumber* and the ClassName has to match the specific label in 
	ImageNet dataset. E.g. `tabby-1.jpg` or `tiger_cat-20.jpg`

3.	Run script `create_csv.py` in subfolder `dataset_manager` to create the necessary csv-file for executing mosaic creation, heatmap calculation
	and evaluation. Provide the dataset name (has to match the folder the data subfolder is in) before running the script.

4. 	Copy the file `imagenet_labels.csv` into the folder.

5. 	Add new dataset in `consts/consts.py` as new class to DatasetArgs and MosaicArgs. Naming has to correspond to dataset folder name.
	Then add it also in `consts/paths.py` to DatasetPaths and MosaicPaths.

6.	Add new dataset in `dataset_manager/datasets.py` as a new dataset class and a new mosaic dataset class (guided by already existing classes).

7. 	Add dataset in `explainability/pipelines.py` under DATASETS guided by already existing entries.

Now the new dataset can be used to run experiments with different saliency methods.

## How to run experiments and visualize results

- To generate a mosaic dataset and corresponding heatmaps plus the classification metrics per mosaic, run script `xai_eval_script.py`, with the arguments
  - **--dataset**: dataset used for mosaic generation
  - **--mosaics**: name of mosaic dataset
  - **--mosaics_per_class**: number of mosaics that will be created per class; if argument is None, existing mosaics will be used, **OTHERWISE EXISTING MOSAICS WILL BE OVERWRITTEN**
  - **--architecture**: classification model to investigate
  - **--xai_method**: saliency method used for generating heatmaps

  e.g. 

  > `python xai_eval_script.py --dataset carsncats --mosaics carsncats_mosaic --mosaics_per_class 10 --architecture resnet50 --xai_method bcos`

  The mosaics will be stored in `data/mosaics/carsncats_mosaic`, the heatmaps will be saved to folder `data/explainability/hash` and the results for the classification metrics will be stored in the corresponding csv-file under `data/explainability/hash.csv`. To find the hash that relates to the experiment, check `data/explainability/hash_explainability.csv`. If the mosaics already exist without a corresponding dataset, simply use the script with a consistent name for the **--dataset** argument and in coherence with the classes mentioned in *Dataset instructions*.

- To receive meaningful results about XAI-performance with the saliency metrics, models should be able to distinguish well between the different classes used in the mosaics. To test this, the script `model_eval.py` can be used. Corresponding accuracies (top-1 and top-5 accuracy for up to three target classes (depending on investigated dataset)) are saved in `evaluation/model_accs.csv`.

- There are a few different ways to vizualize the results of the experiments. For a general inspection of the different heatmaps of a single input image, the `sumgen_script.py` can be used. `evaluation/create_summaries.bat` provides a way as to create all relevant summaries for the datasets used in our paper. Note the `--info` flag when using `sumgen_script.py`, as not using the flag creates summaries as in the paper, whereas `--info` also shows all saliency metrics in the summaries to check whether new metrics work as expected. Created summaries are saved in `data\mosaics\summary`.

  <img src="https://github.com/lelo204/ClassificationMetricsForImageExplanations/blob/main/data/mosaics/summary/imagenet_9981_resnet50.jpg?raw=true" alt="Sample Summary Image for ImageNet" width="700"/>

- To evaluate results over entire datasets, `compute_viz_alphas.py` can be used, where `evaluation\compute_alphas.bat` provides some examples for its usage. This script generates the violin plots of the saliency metric performance, the correlation plots for Spearmans rank correlation as the inter-method reliability (both saved in `evaluation/figures/`) and Krippendorff's $\alpha$ values as the inter-rater reliability (saved as a .csv in `evaluation/alphas.csv`).

- We also evaluated Krippendorff's $\alpha$ between different datasets, different metrics on the same dataset and the two different models (only shortly mentioned in the paper). The script `xai_ranking.py` ranks the saliency methods per architecture and dataset according to the mean and median value for each of the saliency metrics (this is saved in `evaluation/rankings`). With these rankings, Krippendorff's $\alpha$ can be calculated for every architecture - dataset - metric combination (the other experiments can be created with a bit of tweaking in the script). The result is saved in `evaluation/alphas.csv`.

## Citation
Please cite our paper when using this code. 
```
**PLACEHOLDER**
```



## References
<a id="1">[1]</a>
Fresz, B., Lörcher, L., & Huber, M. (2024). Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations. In The 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24). Association for Computing Machinery, New York, NY, USA, 1–19.

<a id="2">[2]</a>
Arias-Duart, A., Parés, F., & García-Gasulla, D. (2021). Focus! Rating XAI Methods and Finding Biases with Mosaics. arXiv preprint arXiv:2109.15035

<a id="3">[3]</a>
Arias-Duart, A., Mariotti, E., Garcia-Gasulla, D., & Alonso-Moral, J. M. (2023). A confusion matrix for evaluating feature attribution methods. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3708-3713).

<a id="4">[4]</a>
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

<a id="5">[5]</a>
Böhle, M., Fritz, M., Schiele, B. (2022).
B-cos Networks: Alignment is all we need for interpretability. arXiv preprint arXiv:2205.10268.

<a id="6">[6]</a>
Krippendorff, K. (2004). Reliability in content analysis: some common misconceptions and recommendations. Human Communication Research, 30(3), 411-433.

<a id="7">[7]</a>
Myers, J. L., Well, A. D., & Lorch Jr, R. F. (2013). Research design and statistical analysis. Routledge.

<a id="8">[8]</a>
Tomsett, R., Harborne, D., Chakraborty, S., Gurram, P., & Preece, A. (2020). Sanity checks for saliency metrics. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 04, pp. 6021-6029).

<a id="9">[9]</a>
Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). 
Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

<a id="10">[10]</a>
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek,
W. (2015). On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one, 10(7), e0130140.

<a id="11">[11]</a>
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).

<a id="12">[12]</a>
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

<a id="13">[13]</a>
Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018, March). Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

<a id="14">[14]</a>
Sundararajan, M., Taly, A., & Yan, Q. (2017, July).
Axiomatic attribution for deep networks. In International Conference on Machine Learning (pp. 3319-3328). PMLR.

<a id="15">[15]</a>
Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., ... & Reblitz-Richardson, O. (2020). Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896.
