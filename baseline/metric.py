import torch
from sklearn.metrics import f1_score


def accuracy(label: torch.tensor, predicted: torch.tensor):
    return (label == predicted).sum().item() / len(label)


def macro_f1(label: torch.tensor, predicted: torch.tensor):
    label, predicted = label.tolist(), predicted.tolist()
    f1_scores = f1_score(y_true=label, y_pred=predicted, average=None)
    return sum(f1_scores) / len(f1_scores)
    # return f1_score(y_true=label, y_pred=predicted, average='macro')
