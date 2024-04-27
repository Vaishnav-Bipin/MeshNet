import numpy as np

def precision_cls(mat, cls):
    n = mat.shape[0]
    TP = mat[cls, cls]
    FP = 0
    for i in range(n):
        if i == cls:
            continue
        FP += mat[cls, i]   # pred = cls, true = not cls
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall_cls(mat, cls):   # This is basically per class accuracy
    n = mat.shape[0]
    TP = mat[cls, cls]
    FN = 0
    for i in range(n):
        if i == cls:
            continue
        FN += mat[i, cls]   # pred = not cls, true = cls
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_cls(mat, cls):
    p = precision_cls(mat, cls)
    r = recall_cls(mat, cls)
    if p + r == 0:
        return 0
    return 2*p*r/(p+r)

def accuracy_avg(mat):
    n = mat.shape[0]

    correct = 0
    total = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                correct += mat[i, j]
            total += mat[i, j]
    return correct / total

def precision_macavg(mat):
    n = mat.shape[0]
    p_avg = 0
    for i in range(n):
        p_avg += precision_cls(mat, i)
        print(str(i) + " " + str(precision_cls(mat, i)))
    p_avg = p_avg / n
    return p_avg

def recall_macavg(mat):
    n = mat.shape[0]
    r_avg = 0
    for i in range(n):
        r_avg += recall_cls(mat, i)
    r_avg = r_avg / n
    return r_avg

def f1_macavg(mat):
    n = mat.shape[0]
    f1_avg = 0
    for i in range(n):
        f1_avg += f1_cls(mat, i)
    f1_avg = f1_avg / n
    return f1_avg
