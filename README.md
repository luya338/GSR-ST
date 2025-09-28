GSR-ST
=======
## Project Overview

GSR-ST is a generalized spatial-temporal deep learning framework for efficiently identifying polyadenylation signals, translation initiation sites, and promoters in DNA sequences. By integrating DNABERT pre-trained embeddings with handcrafted features, it achieves high-accuracy prediction of genomic signals across multiple species.
 
## Requirements
=====
* Linux
* Python 3.8
* TensorFlow=2.9.0


### -----------------------train model----------------------------------

Genomic signal and region datasets including three tasks: PAS, TIS, and Promoters. The dataset used in this study can be downloaded from the (https://www.alipan.com/s/HLzwEd1JiYb).
To train the GSR-ST model, follow these steps:

 1\.Feature Preparation
    Use the code in the "feature" folder to generate the required features.
        ENAC and NCPD features: run "ENAC feature.py" and "NCPD feature.py" respectively;
        DNABERT features:Download the DNABERT pretrained model (https://github.com/jerryji1993/DNABERT);
                         Place the DNABERT-6 model binary file into the 6-new-12w-0 folder and unzip it;
                         Run "k-mer segmentation.py" to split DNA sequences into 6-mers;
                         Run "bert feature.py" to extract DNABERT features.
 2\.Model Training and Evaluation
    PAS and TIS tasks: after loading the required features, run "model1withPAS.py" and "model2withPAS.py", and finally run "model_prediction.py" to train model on PAS and TIS datasets;
    Promoter task: after loading the required features, run "model1withPromoters.py" and "model2withPromoters.py", and finally run "models_prediction.py" to train model on the Promoters dataset.




### -----------------------Predciton----------------------------------

 The genomic signal and region test datasets can be found in the processed_data folder. 
   To re-run predictions using the trained model, run "model test.py" in the test folder for GSR dataset prediction;
   For the PAS independent test set (Hg38), run "Independent test.py" in the test folder.

For questions, please contact: 15998527859@163.com
