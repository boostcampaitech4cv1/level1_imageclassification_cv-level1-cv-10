import torch
from sklearn.metrics import f1_score


def accuracy(label: torch.tensor, predicted: torch.tensor):
    label == predicted

def macro_f1(label: torch.tensor, predicted: torch.tensor):
    label, predicted = label.tolist(), predicted.tolist()
    return f1_score(y_true=label, y_pred=predicted, average='macro')