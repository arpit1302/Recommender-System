import numpy as np


def rmse(Arr, ArrP):
    """
    This function computes Root Mean Square Error.

    Input:
    Arr - Actual numpy array
    ArrP - Predicted numpy array
    Returns: Root Mean square error - float
    """
    x_length = Arr.shape[0]
    y_length = Arr.shape[1]
    error = 0
    N = x_length * y_length
    for x in range(x_length):
        for y in range(y_length):
            error += ((Arr[x][y] - ArrP[x][y]) ** 2) / N
    error = error ** 0.5
    return error



def top_k(k, Arr, ArrP, ignore=True):
    """
    This function returns precision of predicted results in top k ratings.

    Input:
    Arr - Actual numpy array.
    ArrP - Predicted numpy array.
    ignore - Ignores already rated values.

    Returns:
    Precision of predictions in top K - float
    """
    precision = []
    x_length = Arr.shape[0]
    y_length = Arr.shape[1]
    
    for i in range(x_length):
        sorted_M = sorted(Arr[i], reverse=True)[:k]
        for j in range(y_length):
            count = 0
            if ignore:
                if Arr[i][j] == 0:
                    try:
                        if sorted_M.index(ArrP[i][j]) > -1:
                            count += 1
                    except ValueError as ve:
                        pass
            else:
                try:
                    if sorted_M.index(ArrP[i][j]) > -1:
                        count += 1
                except ValueError as ve:
                    pass
        precision.append(count)

    av_precision = 0
    p_len = len(precision)
    for p in precision:
        av_precision += p / p_len
    return av_precision

def spearman_correlation(Arr, ArrP):
    """
    This function returns Spearman score for the prediction.
    Formula: 1 - [sum(diff(predicted - actual)^2) / n((n^2)-1)]

    Input:
    Arr- Actual numpy array.
    ArrP - Predicted numpy array.

    Returns:
    Spearman score - float
    """
    x_length = Arr.shape[0]
    y_length = Arr.shape[1]
    
    s = 0
    N = 0
    
    for i in range(x_length):
        for j in range(y_length):
            if Arr[i][j] != 0:
                N += 1
                s += (Arr[i][j] - ArrP[i][j]) ** 2
    N = (N*(N**2 - 1))
    s = 1 - (s/N)
    return s
