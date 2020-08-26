from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def evaluate(y, yhat):
    return confusion_matrix(y, yhat), \
           precision_score(y, yhat), \
           recall_score(y, yhat), \
           f1_score(y, yhat)




