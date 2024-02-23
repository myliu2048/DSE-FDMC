## Progressive Neighbor-masked Contrastive Learning for Fusion-style Deep Multi-view Clustering

### Requirements

python==3.8.5

pytorch==1.7.1+cu110

numpy==1.21.2

scikit-learn==0.23.2

### Datasets

MNIST_USPS and Hdigit are placed in "data" folder, Prokaryotic, Caltech-5V, and YouTubeFace can be downloaded from [dropbox](https://www.dropbox.com/scl/fi/shv57hvaq1xo7teurt6xh/data.zip?rlkey=lo05a9iegemt9tuhpe5osj1du&dl=0), unzip all the data, put in 'data/' folder.

### Usage


- To train the network and save model, run:

```bash
python main.py
```

- To test the trained model, download the trained models from [dropbox](https://www.dropbox.com/scl/fi/dn69vhun6rnfu9xieto2e/models.zip?rlkey=t7lxliqobh145i43n4jpnkd5k&dl=0), and unzip all the model named by corresponding data, put in 'models/' folder, and run: 
```bash
python test.py
```

**Note: Due to Pytorch built-in TransformerEncoderLayer does not output attention matrix by default, you should make some changes, please refer to [GCFAgg](https://github.com/Galaxy922/GCFAggMVC/blob/main/Obtain%20-S.docx).**

### Acknowledgments

Work and Code are inspired by [GCFAgg](https://github.com/Galaxy922/GCFAggMVC), [MFLVC](https://github.com/SubmissionsIn/MFLVC), [CONAN](https://github.com/Guanzhou-Ke/conan), [CoMVC](https://github.com/DanielTrosten/mvc) ... 



