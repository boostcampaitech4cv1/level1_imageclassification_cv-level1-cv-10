import torch
from sklearn.metrics import f1_score, confusion_matrix, recall_score


def accuracy(label: torch.tensor, predicted: torch.tensor):
    return (label == predicted).sum().item() / len(label)


def micro_acc(label: torch.tensor, predicted: torch.tensor):
    label, predicted = label.tolist(), predicted.tolist()
    matrix = confusion_matrix(label, predicted)
    print(matrix.diagonal() / matrix.sum(axis=1))


def macro_f1(label: torch.tensor, predicted: torch.tensor, return_all=False):
    label, predicted = label.tolist(), predicted.tolist()
    f1_scores = f1_score(y_true=label, y_pred=predicted, average=None)

    if return_all:
        return f1_scores
    return sum(f1_scores) / len(f1_scores)
    # return f1_score(y_true=label, y_pred=predicted, average='macro')


if __name__ == "__main__":
    tmp = torch.rand(10)*2
    print(tmp)
    tmp[tmp>=1] = 1
    tmp[tmp<1] = 0
    print(tmp)
