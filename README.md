# FragDPI

## Title ##
FragDPI: A Novel Drug–Protein Interaction Prediction Model Based on Fragment Understanding and Unified Coding

## Abstract ##
we propose a deep learning model using the Bidirectional Transformers as the backbone, FragDPI, to predict the Drug-Protein binding affinity. Unlike other methods, we encode the sequences only based on the conserved fragments, and encode the protein and drug into a unified vector.  
Moreover, we adopt a novel two-step training strategy to build FragDPI. The pre-training step is to learn the interaction between different fragments with unsupervised learning, and the fine-tuning step is for predicting the binding affinity with supervised learning. The comprehensive experiment results have illustrated the superiority of our proposed FragDPI. 
## How to use ##
Use the following command to pretrain the model:
 ```shell
    sh pretrain.sh
```
Use the following command to fine-tune the model:
 ```shell
    sh train.sh
```
Use the following command to predict the binding affinity score:
 ```shell
    sh test.sh
```