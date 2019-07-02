# Author: GC

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from preprocess import load_obj


class NERDataset(Dataset):
    """NER Dataset
    """

    def __init__(self, dataset_pkl):
        super(NERDataset, self).__init__()
        self.dataset = load_obj(dataset_pkl)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.dataset[idx][0], dtype=torch.long),
            torch.tensor(self.dataset[idx][1], dtype=torch.long),
        )


class BatchPadding(object):
    """Padding in batch and sequences is sorted by length in order
    """

    def __init__(self, descending=True):
        self.reverse = True if descending else False

    def __call__(self, batch):
        """batch should be a list of tensors
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=self.reverse)
        seqs, tags = tuple(zip(*sorted_batch))
        seqs = pad_sequence(seqs) # 0 padding
        tags = pad_sequence(tags)
        masks = seqs.ne(0)
        return seqs, tags, masks # (seq_len, batch_size)


if __name__ == "__main__":

    dataset = NERDataset("data/msra/processed/train.pkl")
    print(len(dataset))

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=10, collate_fn=BatchPadding())
    for seqs, tags, masks in loader:
        print(seqs)
        break