# Author: GC

import argparse
import os

import torch

from consts import UNK
from model import BiLSTM_CRF
from preprocess import load_obj

parser = argparse.ArgumentParser()
parser.add_argument("--sentence", type=str, default=None)
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()


def tagging(model, sentence, ix_to_tag):
    """Do named entity recognition
    """
    sentence = sentence.unsqueeze(1)
    mask = sentence.ne(0)
    best_tag_ids = model.decode(sentence, mask)
    tags = [ix_to_tag[idx] for idx in best_tag_ids[0]]
    return tags


if __name__ == "__main__":

    if args.sentence is None:
        raise ValueError("Please input an sentence")
    if args.model is None:
        raise ValueError("Please specify model file path")

    data_dir = "data/msra/processed"
    word_to_ix = load_obj(os.path.join(data_dir, "word_to_ix.pkl"))
    tag_to_ix = load_obj(os.path.join(data_dir, "tag_to_ix.pkl"))

    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    # Load trained model
    model = BiLSTM_CRF(len(word_to_ix), len(tag_to_ix), 100, 200, 0.1)
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    model.eval()

    # Predict
    sentence = torch.LongTensor(
        [word_to_ix.get(w, word_to_ix[UNK]) for w in args.sentence]
    )
    best_tags = tagging(model, sentence, ix_to_tag)
    print(" ".join(best_tags))
