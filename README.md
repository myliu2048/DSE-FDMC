## Progressive Contrastive Learning for Fusion-style Deep Multi-view Clustering
 
<!-- ### Authors -->
<!-- Minyang Liu, Zuyuan Yang, Zhenni Li, Shengli Xie -->

<!-- This repo contains the code and data of our CVPR'2024 paper [Progressive Contrastive Learning for Fusion-style Deep Multi-view Clustering](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.pdf). -->


<!-- ## 1. Framework

<img src="fig/framework.png"  width="897" height="317" />

 The framework of PCL-FDMC. We first employ the autoencoders with MSE reconstruction loss to learn low-level view-specific embeddings $\mathbf{Z}^v$ from raw multi-view data. We then concatenate the embeddings from individual views and perform locally consistent aggregation on concatenated representation $\mathbf{Z}$ to obtain the fusion representation $\hat{\mathbf{Z}}$. Low-level embeddings $\mathbf{Z}^v$ and the fusion representation $\hat{\mathbf{Z}}$ are further mapped into the high-level  space to get $\mathbf{H}^v$ and enhanced fusion representation $\hat{\mathbf{H}}$. MSE denotes Mean Square Error loss; TEL denotes Transformer Encoder Layer; MLP denotes Multi-Layer Perception; NMCL denotes Neighbor-Masked Contrastive Loss. -->

### Requirements

python==3.8.5

pytorch==1.7.1+cu110

numpy==1.21.2

scikit-learn>=0.23.2

### Datasets

All datasets are placed in "data" folder. 

### Usage

The code includes:

- train a new model：

```bash
python main.py
```

- test the trained model:
download the trained models from [dropbox](https://www.dropbox.com/scl/fi/dn69vhun6rnfu9xieto2e/models.zip?rlkey=t7lxliqobh145i43n4jpnkd5k&dl=0), and unzip to models, 
```bash
python test.py
```


### Acknowledgments

Work and Code are inspired by [GCFAgg](https://github.com/Galaxy922/GCFAggMVC), [MFLVC](https://github.com/SubmissionsIn/MFLVC), [CONAN](https://github.com/Guanzhou-Ke/conan), [CoMVC](https://github.com/DanielTrosten/mvc) ... 

<!-- ### Citation

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{Yan_2023_CVPR,
    author    = {Yan, Weiqing and Zhang, Yuanyang and Lv, Chenlei and Tang, Chang and Yue, Guanghui and Liao, Liang and Lin, Weisi},
    title     = {GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19863-19872}
}
``` -->

<!-- If you have any problems, contact me via myliu2048@gmail.com. -->

