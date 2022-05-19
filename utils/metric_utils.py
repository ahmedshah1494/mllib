import torch

def get_preds_from_logits(logits):
    if (logits.dim() == 1) or ((logits.dim() == 2) and (logits.shape[1] == 1)):
        pred = (logits > 0.5).float()
    else:
        pred = torch.argmax(logits, dim=1)
    return pred

def compute_accuracy(logits, y):
    pred = get_preds_from_logits(logits)
    correct = (pred == y)
    acc = correct.sum().float() / len(y)
    return float(acc), correct