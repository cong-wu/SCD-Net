# SCD-Net [AAAI2024]
The Official implementation for  'SCD-Net: Spatiotemporal Clues Disentanglement Network for Self-supervised Skeleton-based Action Recognition' (AAAI 2024). [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28409)  [arXiv](https://arxiv.org/abs/2309.05834)  



- [Prerequisite](#Prerequisite)
- [Data](#Data)
- [Training&Testing](#Training&Testing)
- [Log files](#Log)


<a name="Prerequisite"></a>

# Prerequisite

- Pytorch

- We provided requirement file to install all packages, just by running


`pip install -r requirements.txt`

 
<a name="Data"></a>

# Data

## Generate the  data 

**Download the raw data**

- [NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/). 
- [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).

**Preprocess**

- Preprocess data with `python ntu_gendata.py`.


<a name="Training&Testing"></a>

# Training&Testing

## Training 

- To train on NTU-RGB+D 60 under Cross-Subject evaluation, you can run


    `python ./pretraining.py --lr 0.01 --batch-size 64 --encoder-t 0.2   --encoder-k 8192 
                --checkpoint-path ./checkpoints/pretrain/ 
                --schedule 351  --epochs 451  --pre-dataset ntu60 
                --protocol cross_subject --skeleton-representation joint`

## Testing 


- For action recognition on NTU-RGB+D 60 under Cross-Subject evaluation, you can run


    `python ./action_classification.py --lr 2 --batch-size 1024 
                --pretrained ./checkpoints/pretrain/checkpoint.pth.tar 
                --finetune-dataset ntu60 --protocol cross_subject --finetune_skeleton_representation joint`

- For action retrieval on NTU-RGB+D 60 under Cross-Subject evaluation, you can run


    `python ./action_retrieval.py --knn-neighbours 1 
                --pretrained ./checkpoints/pretrain/checkpoint.pth.tar 
                --finetune-dataset ntu60 --protocol cross_subject --finetune-skeleton-representation joint`

<a name="Log"></a>

# Log files

We also provided some the testing logs in ./log
