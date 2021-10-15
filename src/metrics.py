
def f1_at_k(pred, gold, k = 20):

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
    

def precision_at_k(pred, gold, k = 20):

    if len(pred) > k:
        pred = pred[:k]

    tp = len(set(pred) & set(gold))
    fp = len(set(pred) - set(gold))

    precision = tp/(tp + fp)
    return precision


def recall_at_k(pred, gold, k = 20):

    if len(pred) > k:
        pred = pred[:k]

    tp = len(set(pred) & set(gold))
    fn = len(set(gold) - set(pred))

    recall = tp/(tp + fn)
    return recall