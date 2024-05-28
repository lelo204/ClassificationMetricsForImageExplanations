# Classification Metrics for Image Explanations

In [[1]](#1) we further develop the work of [ref] and [ref], which provide a set of evaluation metrics for saliency methods. We extend this set to a comprehensive list of metrics that mimic common metrics for classification evaluation based on the definition of correct and incorrect feature importance in images. Particularly, the following points are addressed:

- We include saliency metrics that produce interesting results (e.g. specificity), but where overlooked in [ref]
- In addition to the saliency methods discussed in [ref], we also include SHAP [ref] and Bcos [ref]
- We show how such metrics can be evaluated using Krippendorff's $\alpha$ [ref] and Spearman's $\rho$ [ref] instead of taking them at face value (which is already a problem in XAI, as discussed in the paper)

This repository contains the code and datasets that are needed to recreate the experiments conducted in our paper: [Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations](LINK). As our paper is an extension of [[2]](#2), this repository also builds on their implementation of the [Focus-metric](https://github.com/HPAI-BSC/Focus-Metric).

### Explainability Methods: 

- Smoothgrad [[10]](#10); implementation based on this [repo](https://github.com/wjNam/Relative_Attributing_Propagation) of Nam *et al.* 
- Layer-wise Relevance Propagation (LRP) [[2]](#2); implementation based on this [repo](https://github.com/kazuto1011/grad-cam-pytorch) of Nakashima *et al.*
- GradCAM [[9]](#9); implementation based on this [repo](https://github.com/jacobgil/pytorch-grad-cam)
of Gildenblat *et al.*
- LIME [[7]](#7); implementation based on this [repo](https://github.com/marcotcr/lime) of Tulio *et al.*
- GradCAM++ [[3]](#3); implementation based on this [repo](https://github.com/jacobgil/pytorch-grad-cam) of Gildenblat *et al.*
- Integrated Gradients (IG) [[11]](#11); implementation based on this [repo](https://github.com/pytorch/captum) of Kokhlikyan *et al.* [[4]](#4).
- KernelSHAP by [Captum](https://captum.ai/api/kernel_shap.html) as a representative of the SHAP-family
- BCos-Networks version 2 based on the [work](https://github.com/B-cos/B-cos-v2) of Boehle *et al.* [[12]](#12).

The first six sailiency methods have been adopted unchanged from the [Focus-metric](https://github.com/HPAI-BSC/Focus-Metric) repository. The last two (KernelSHAP and BCos) were added by us.


### Requirements

This code runs under Python 3.10.4. The python dependencies are defined in `requirements.txt`. 


## Available mosaics

We provide two data sets (located in folder _data_) that can be used to generate mosaics for the XAI evaluation (this is done by executing the script `xai_eval_scripy.py`; detailed instructions on how to run experiments see below). The authors of [[2]](#2) provide mosaics in their [repository](https://github.com/HPAI-BSC/Focus-Metric#requirements), which can also be used with our code. To do so, download the mosaics and copy them to `data\mosaics\`. When executing `xai_eval_script.py` the '--dataset' argument has to correspond to the mosaics (i.e. --dataset ilsvrc2012 for --mosaics ilsvrc2012_mosaics) and the '--mosaics_per_class' argument has to be None.

### Dataset Instructions

The models used here are all trained on ImageNet and are not fine-tuned. Therefore, new data sets should consist of classes that are available in ImageNet. To create a new data set so that it can be used with the code, the following steps must be performed:

1. 	Create subfolder in 'data/datasets/' with the dataset name.

2. 	Add folder 'data', which contains images that will be used for mosaic creation. 
	Images should be named 'ClassName-ImageNumber' and the ClassName has to match the specific label in 
	ImageNet dataset. E.g. 'tabby-1.jpg' or 'tiger_cat-20.jpg'

3.	Run script create_csv.py in subfolder 'dataset_manager' to create the necessary csv-file for executing mosaic creation, heatmap calculation
	and evaluation. Provide the dataset name (has to match the folder the data subfolder is in) before running the script.

4. 	Copy the file 'imagenet_labels.csv' into the folder.

5. 	Add new dataset in 'consts/consts.py' as new class to DatasetArgs and MosaicArgs. Naming has to correspond to dataset folder name.
	Then add it also in 'consts/paths.py' to DatasetPaths and MosaicPaths.

6.	Add new dataset in 'dataset_manager/datasets.py' as a new dataset class and a new mosaic dataset class (guided by already existing classes).

7. 	Add dataset in 'explainability/pipelines.py' under DATASETS guided by already existing entries.



## Dataset Instructions

1. 	Create subfolder in 'data/datasets/' with the dataset name.

2. 	Add folder 'data', which contains images that will be used for mosaic creation. 
	Images should be named 'ClassName-ImageNumber' and the ClassName has to match the specific label in 
	ImageNet dataset. E.g. 'tabby-1.jpg' or 'tiger_cat-20.jpg'

3.	Run script create_csv.py in subfolder 'dataset_manager' to create the necessary csv-file for executing mosaic creation, heatmap calculation
	and evaluation. Provide the dataset name (has to match the folder the data subfolder is in) before running the script.

4. 	Copy the file 'imagenet_labels.csv' into the folder.

5. 	Add new dataset in 'consts/consts.py' as new class to DatasetArgs and MosaicArgs. Naming has to correspond to dataset folder name.
	Then add it also in 'consts/paths.py' to DatasetPaths and MosaicPaths.

6.	Add new dataset in 'dataset_manager/datasets.py' as a new dataset class and a new mosaic dataset class (guided by already existing classes).

7. 	Add dataset in 'explainability/pipelines.py' under DATASETS guided by already existing entries.

8. 	Done! Now you can run the script 'xai_eval_script.py' with your new dataset.
	Example:

	python xai_eval_script.py --dataset carsncats --mosaics carsncats_mosaic --mosaics_per_class 10 --architecture resnet50 --xai_method bcos



## How to run the experiments

We already provide the bash scripts needed to calculate the focus of the 
different settings. Each execution has two steps:

1. First, the explainability method is applied and the relevance matrices
   are obtained and save in: 
   ```$PROJECT_PATH/data/explainability/```

2. Second, the _Focus_ is computed from the relevances obtained in the previous step.
   
To run both steps execute the following bash scripts:

#### Step 1
> `cd $PROJECT_PATH/explainability/scripts/explainability_dataset/`
     
> `sh explainability_dataset_architecture_method.sh`

#### Step 2
> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_dataset_architecture_method.sh`

where:
  * **dataset** must be exchanged by 
  one of the following options: **catsdogs**, **ilsvrc2012**, **mit67** or **mame**. 
  * **architecture** must be exchanged by one of the following options: **alexnet**, **vgg16** or **resnet18**.
  * **method** must be exchanged by **smoothgrad**, **lrp**, **gradcam**, **lime**, **gradcampp** or **intgrad**.


For example, to get the _Focus_ of the Dogs vs. Cats dataset,
using the ResNet18 architecture and the GradCAM method,
run the following:

#### Step 1
> `cd $PROJECT_PATH/explainability/scripts/explainability_catsdogs/`

> `sh explainability_catsdogs_resnet18_gradcam.sh`

#### Step 2
> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_catsdogs_resnet18_gradcam.sh`

## Cite
Please cite our paper when using this code. 
```
@misc{ariasduart2021focus,
      title={Focus! Rating XAI Methods and Finding Biases with Mosaics}, 
      author={Anna Arias-Duart and Ferran Parés and Dario Garcia-Gasulla},
      year={2021},
      eprint={2109.15035},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



## References
<a id="1">[1]</a>
Fresz, B., Lörcher, L., & Huber, M. (2024). Classification Metrics for Image Explanations: Towards Building Reliable XAI-Evaluations.

<a id="2">[2]</a>
Arias-Duart, A., Parés, F., & García-Gasulla, D. (2021). Focus! Rating XAI Methods and Finding Biases with Mosaics. arXiv preprint arXiv:2109.15035


<!-- <a id="2">[2]</a>
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek,
W. (2015). On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one, 10(7), e0130140. -->

<a id="3">[3]</a>
Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018, March). Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

<a id="4">[4]</a>
Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., ... & Reblitz-Richardson, O. (2020). Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896.


<a id="5">[5]</a>
Parés, F., Arias-Duart, A., Garcia-Gasulla, D., Campo-Francés, G., Viladrich, N.,
Ayguadé, E., & Labarta, J. (2020). A Closer Look at Art Mediums: 
The MAMe Image Classification Dataset. arXiv preprint arXiv:2007.13693.

<a id="6">[6]</a>
Quattoni, A., & Torralba, A. (2009, June). Recognizing indoor scenes. 
In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 413-420). 
IEEE.

<a id="7">[7]</a>
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

<a id="8">[8]</a>
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei,
L. (2015). Imagenet large scale visual recognition challenge. International journal
of computer vision, 115(3), 211-252.


<a id="9">[9]</a>
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).

<a id="10">[10]</a>
Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). 
Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

<a id="11">[11]</a>
Sundararajan, M., Taly, A., & Yan, Q. (2017, July).
Axiomatic attribution for deep networks. In International Conference on Machine Learning (pp. 3319-3328). PMLR.

<a id="12">[12]</a>
Böhle, M., Fritz, M., Schiele, B. (2022).
B-cos Networks: Alignment is All we Need for Interpretability. arXiv preprint arXiv:2205.10268.