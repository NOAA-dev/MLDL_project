import numpy as np


def compute_confusion_matrix(y_true, y_pred, num_classes):
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= num_classes):
        raise ValueError("Class IDs must be between 0 and num_classes - 1.")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def normalize_rows(cm):
    
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)


def top_confusions(cm, class_names, top_n=10):

    rows = []
    for true_id in range(cm.shape[0]):
        class_total = cm[true_id].sum()
        for pred_id in range(cm.shape[1]):
            count = int(cm[true_id, pred_id])
            if true_id != pred_id and count > 0:
                rows.append({
                    "true": class_names[true_id],
                    "predicted": class_names[pred_id],
                    "count": count,
                    "share_of_true_class": count / class_total,
                })
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:top_n]