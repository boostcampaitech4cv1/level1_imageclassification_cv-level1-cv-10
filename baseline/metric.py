import torch
from sklearn.metrics import f1_score,confusion_matrix,recall_score


def accuracy(label: torch.tensor, predicted: torch.tensor):
    return (label == predicted).sum().item() / len(label)


def micro_acc(label: torch.tensor, predicted: torch.tensor):
    label, predicted = label.tolist(), predicted.tolist()
    matrix  = confusion_matrix(label, predicted)
    print(matrix.diagonal()/matrix.sum(axis=1))


def macro_f1(label: torch.tensor, predicted: torch.tensor,return_all=False):
    label, predicted = label.tolist(), predicted.tolist()
    f1_scores = f1_score(y_true=label, y_pred=predicted, average=None)

    if return_all:
        return f1_scores 
    return sum(f1_scores) / len(f1_scores)
    # return f1_score(y_true=label, y_pred=predicted, average='macro')

if __name__ == "__main__":
    import numpy as np
    f1_all = np.array([0.3,0.66,0.89,0.77,0.2])
    print(dict(sorted([(i,round(f1_all[i],3)) for i in range(len(f1_all))],key=lambda x:x[1])[:5]))