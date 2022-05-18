import torch


def compute_accuracy(logits, y):
    if (logits.dim() == 1) or ((logits.dim() == 2) and (logit.shape[1] == 1)):
        pred = (logits > 0.5).float()
    else:
        pred = torch.argmax(logits, dim=1)
    correct = (pred == y)
    acc = correct.sum().float() / len(y)
    return float(acc), correct