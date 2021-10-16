
def mean_f1_at_k(list_pred, list_gold, k) -> float:
    acc = 0
    for i, pred in enumerate(list_pred):
        f1 = f1_at_k(pred, list_gold[i], k = k)
        acc += f1
    return acc/len(list_pred)

def f1_at_k(pred, gold, k) -> float:

    if len(pred) > k:
        pred = pred[:k]

    tp = len(set(pred) & set(gold))
    fp = len(set(pred) - set(gold))
    fn = len(set(gold) - set(pred))

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    #exception precision and recall == 0
    try:
        f1_score = 2* (precision * recall)/(precision + recall)
    except:
        return 0.0
    return f1_score


def precision_at_k(pred, gold, k) -> float:

    if len(pred) > k:
        pred = pred[:k]

    tp = len(set(pred) & set(gold))
    fp = len(set(pred) - set(gold))

    precision = tp/(tp + fp)
    return precision


def recall_at_k(pred, gold, k):

    if len(pred) > k:
        pred = pred[:k]

    tp = len(set(pred) & set(gold))
    fn = len(set(gold) - set(pred))

    recall = tp/(tp + fn)
    return recall

