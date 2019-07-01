# Author: GC

import argparse
import os

import torch
import torch.optim as optim
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataset import BatchPadding, NERDataset
from model import BiLSTM_CRF
from preprocess import load_obj

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--dataset", type=str, default="Chinese")
parser.add_argument("--embed-dim", type=int, default=100)
parser.add_argument("--hidden-dim", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args()

os.makedirs(f"checkpoints/{args.name}", exist_ok=True)

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score(y_true, y_pred):
    """Wrapper of seqeval metrics
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def evaluate(model, loader, ix_to_tag):
    """Evaluate on dev or test data
    """
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for seqs, tags, masks in loader:
            # pred
            tags_pred = model.decode(seqs.to(device), masks.to(device))
            for tp in tags_pred:
                y_pred.append([ix_to_tag[ix] for ix in tp])
            # true
            lens = masks.sum(0).tolist()
            tags_l = tags.t().tolist()
            for t, ln in zip(tags_l, lens):
                y_true.append([ix_to_tag[ix] for ix in t[:ln]])
        score = score(y_true, y_pred)
    return score


if __name__ == "__main__":
    data_dir = f"data/{args.dataset}/processed"

    # Load dataset
    train_data = NERDataset(os.path.join(data_dir, "train.pkl"))
    test_data = NERDataset(os.path.join(data_dir, "test.pkl"))
    dev_data = NERDataset(os.path.join(data_dir, "dev.pkl"))

    # Load vocabs
    word_to_ix = load_obj(os.path.join(data_dir, "word_to_ix.pkl"))
    tag_to_ix = load_obj(os.path.join(data_dir, "tag_to_ix.pkl"))

    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    # DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=BatchPadding(),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        collate_fn=BatchPadding(),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=BatchPadding(),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = BiLSTM_CRF(
        len(word_to_ix), len(tag_to_ix), args.embed_dim, args.hidden_dim, args.dropout
    ).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Training...")
    best_dev_f1 = 0
    bad_count = 0
    for epoch in range(args.epochs):
        model.train()
        for i, (seqs, tags, masks) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = model.loss(seqs.to(device), tags.to(device), masks.to(device))
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                print(
                    "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        i * seqs.size(1),
                        len(train_loader.dataset),
                        100.0 * i / len(train_loader),
                        loss.item(),
                    )
                )
        print("Evaluating...")
        dev_precision, dev_recall, dev_f1 = evaluate(model, dev_loader, ix_to_tag)
        test_precision, test_recall, test_f1 = evaluate(model, test_loader, ix_to_tag)
        print(f"\ndev\tprecision: {dev_precision}, recall: {dev_recall}, f1: {dev_f1}")
        print(f"test\tprecision: {test_precision}, recall: {test_recall}, f1: {test_f1}\n")

        torch.save(model.state_dict(), f"checkpoints/{args.name}/model-epoch{epoch}.pt")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= args.patience:
                print("Early stopped!")
                break
    print("Best dev F1: ", best_dev_f1)
