# ner_bilstm_crf

## Requirements

* pytorch >= 1.0.0
* pytorch-crf >= 0.7.2
* seqeval >= 0.0.12

## Dataset

Create a 'data/Chinese' folder and place train.txt/dev.txt/test.txt into it, you can download them from <https://github.com/GeneZC/Chinese-NER/tree/master/data>

Other datasets are also applicable if they are organized according to the format below:

小 B-PER  
明 I-PER  
住 O  
在 O  
北 B-LOC  
京 I-LOC

联 B-ORG  
合 I-ORG  
国 I-ORG

and place data files in a subfolder named the dataset name under data folder

## Preprocess

Run 'python preprocess.py --dataset=[dataset-name]', then processed data will be generated and be placed in 'data/[dataset-name]/processed' folder, including transformed train/dev/test data and some vocabularies.

## Train

Run 'python train.py --help' to get some training settings, during training, model performances on dev and test dataset is printed every epoch, including Precision, Recall and F1. Also a checkpoint model file is saved at the end of every epoch.

Training process will run on CUDA by default if supported.

Example:

```bash
python train.py --name="name-of-this-train" --dataset="dataset-name"
```

## Tagging

Tagging using trained model, example:

```bash
python tagging.py --sentence="小明住在北京" --model="checkpoints/train-name/model-epochX.pt"
```
