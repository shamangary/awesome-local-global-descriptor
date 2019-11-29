# awesome-local-global-descriptor
This is my personal note about local and global descriptor. Trying to make anyone can get in to these fields more easily.
If you find anything you want to add, feel free to post on issue or email me.

***This repo will be constantly updated.***

***Author: Tsun-Yi Yang (shamangary@hotmail.com)***


# Local matching pipeline

In this section, I focus on the review about the sparse keypoint matching and it's pipeline.

### 1. Keypoint detection
This subsection includes the review about keypoint detection and it's orientation, scale, or affine transformation estimation.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
|[ICCV19]|  Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters   |  [PDF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Barroso-Laguna_Key.Net_Keypoint_Detection_by_Handcrafted_and_Learned_CNN_Filters_ICCV_2019_paper.pdf) |  [Github](https://github.com/axelBarroso/Key.Net)  |
|[CVPR17]| Quad-networks: unsupervised learning to rank for interest point detection | [PDF](https://inf.ethz.ch/personal/ladickyl/quad_cvpr17.pdf)| - |

### 2. Keypoint description (local descriptor)
In the last few decades, people focus on the patch descriptor

+ Hand-crafted

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |

+ Deep learning

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |

### 3. End-to-end matching pipeline
Recently, more and more papers try to embed the whole matching pipeline into one framework.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| [CVPR19] |  D2-Net: A Trainable CNN for Joint Description and Detection of Local Features   |  [arXiv](https://arxiv.org/pdf/1905.03561),[Page](https://dsmn.ml/publications/d2-net.html)  |  [Github](https://github.com/mihaidusmanu/d2-net)  |

### 4. Geometric verification
After the matching, standard RANSAC and it's variants are usually adopted for outlier removal.

+ Hand-crafted

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |

+ Deep learning

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |


# Global retrieval

Consider global retrieval usually targets on a lot of candidates, there are several way to generate one single description for one image.

### 1. Local feature aggregation
When there is only hand-crafted local descriptors, people usually uses feature aggregation from a set of local descriptors and output a single description.


### 2. Real-valued descriptor
One single representation from the image.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| [TPAMI18] |   Fine-tuning CNN Image Retrieval with No Human Annotation  |   [arXiv](https://arxiv.org/abs/1711.02512)  |  [Github](https://github.com/filipradenovic/cnnimageretrieval-pytorch)  |
| [ECCV16] |   CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples  |   [arXiv](https://arxiv.org/abs/1604.02426) |  [Github](https://github.com/filipradenovic/cnnimageretrieval-pytorch)  |

### 3. Binary descriptor
For more compact representation, a binary descriptor can be generated from hashing or thresholding.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |


### 4. Product quantization
A very popular trend of effiecient and compact representation retrieval.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |



# Multi-tasking local and global network
Some works try to cover both local descriptor and global retrieval due to the shared similarity about the activation and the applications.

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| [CVPR19] |   From Coarse to Fine: Robust Hierarchical Localization at Large Scale with HF-Net  |   [arXiv](https://arxiv.org/pdf/1812.03506)  |  [Github](https://github.com/ethz-asl/hfnet)  |


# Benchmarks

| Year | Paper | link | Code |
| --- | --- | --- | --- |
| |     |     |     |

