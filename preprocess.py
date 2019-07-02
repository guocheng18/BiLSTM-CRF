# Author: GC

import argparse
import os
import pickle

from consts import PAD, UNK


word_to_ix = {PAD: 0, UNK: 1}

tag_to_ix = {}


def is_number(s):
    """Determine if the string is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_file(filename):
    """Parse plain text data into arrays
    """
    with open(filename, "r", encoding="utf8") as f:
        sentences = f.read().split("\n\n")
        data = []
        for s in sentences:
            tuples = [ln.split() for ln in s.split("\n")]
            data.append(tuple(zip(*tuples)))
    return data


def fill_vocabs(train_data):
    """Extract all words in training data
    """
    global word_to_ix
    global tag_to_ix
    for words, tags in train_data:
        for w in words:
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)


def replace_digits_with_zero(data):
    """Follow the paper's implementation
    """
    new_data = []
    for words, tags in data:
        new_words = []
        for w in words:
            new_w = "0" if is_number(w) else w
            new_words.append(new_w)
        new_tags = list(tags)
        new_data.append((new_words, new_tags))
    return new_data


def mask_randomly(train_data, percentage):
    """Randomly masks some words in training data
        to train <UNK> tag
    """
    raise NotImplementedError


def dump_obj(obj, filename):
    """Wrapper of pickle.dump
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """Wrapper of pickle.load
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def transform(data):
    """Transform words and tags to ids
    """
    new_data = []
    unknown_word_count = 0
    total_word_count = 0
    for words, tags in data:
        word_ids = [word_to_ix.get(w, word_to_ix[UNK]) for w in words]
        tag_ids = [tag_to_ix.get(t) for t in tags]
        new_data.append((word_ids, tag_ids))
        # count
        total_word_count += len(words)
        for w in words:
            if w not in word_to_ix:
                unknown_word_count += 1
    unknown_proportion = unknown_word_count / total_word_count
    return new_data, unknown_proportion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msra")
    args = parser.parse_args()

    # Example: [(('I', 'am', 'in', 'Beijing'), ('B-PER', 'O', 'O', 'B-LOC')), ...]
    train_data = read_file(f"data/{args.dataset}/raw/train.txt")
    test_data = read_file(f"data/{args.dataset}/raw/test.txt")
    dev_data = read_file(f"data/{args.dataset}/raw/dev.txt")
    print(train_data[0], test_data[0], dev_data[0])
    print(f"train: {len(train_data)}, test: {len(test_data)}, dev: {len(dev_data)}")

    train_data = replace_digits_with_zero(train_data)
    test_data = replace_digits_with_zero(test_data)
    dev_data = replace_digits_with_zero(dev_data)
    print(train_data[0], test_data[0], dev_data[0])

    fill_vocabs(train_data)
    print(f"words: {len(word_to_ix)}")
    print(tag_to_ix)

    # Transform
    train_data, _ = transform(train_data)
    test_data, p1 = transform(test_data)
    dev_data, p2 = transform(dev_data)
    print(train_data[0], test_data[0], dev_data[0])
    print(f"Unknown proportion, dev: {p2}, test: {p1}")

    # Save
    data_dir = f"data/{args.dataset}/processed"
    dump_obj(train_data, os.path.join(data_dir, "train.pkl"))
    dump_obj(test_data, os.path.join(data_dir, "test.pkl"))
    dump_obj(dev_data, os.path.join(data_dir, "dev.pkl"))
    dump_obj(word_to_ix, os.path.join(data_dir, "word_to_ix.pkl"))
    dump_obj(tag_to_ix, os.path.join(data_dir, "tag_to_ix.pkl"))
    print("Done!")
