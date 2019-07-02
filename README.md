# ner_bilstm_crf

## Requirements

* python  = 3.6
* pytorch = 1.0.0
* pytorch-crf = 0.7.2
* seqeval = 0.0.12

## Dataset

First prepare the following directories for each dataset,

* data/[dataset-name]/raw
* data/[dataset-name]/processed

then place `train.txt, dev.txt and test.txt` into the raw folder, note data is supposed to be organized as that
one word/tag one line, sentences are sepreated by blankline.

For Chinese NER dataset, you can access MSRA from <https://github.com/GeneZC/Chinese-NER/tree/master/data>  
For English NER dataset, you can access CoNLL03 from <https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003>

## Preprocess

Run preprosess.py to transform the raw data into processed one, which includes transformed dataset and vocabularies for words and tags, and the processed data
will be placed in `data/[dataset-name]/processed` folder, for example:

```bash
python preprocess.py --dataset="dataset-name"
```

## Train

Run `python train.py --help` to get some training settings, during training, model performances on dev and test dataset is printed every epoch, including Precision, Recall and F1. Besides a model checkpoint file will be saved at the end of every epoch.

Training process will run on GPU by default.

Example:

```bash
python train.py --name="name-of-train" --dataset="dataset-name"
```

## Tagging

Tagging using trained model.

Example:

```bash
python tagging.py --sentence="中国同加利福尼亚州的友好交往源远流长" --model="checkpoints/name-of-train/model-epochX.pt"
```

Output:

```bash
B-LOC I-LOC O B-LOC I-LOC I-LOC I-LOC I-LOC I-LOC O O O O O O O O O
```
