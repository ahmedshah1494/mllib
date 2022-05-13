import torch


def compute_accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    correct = (pred == y)
    acc = correct.sum().float() / len(y)
    return float(acc), correct