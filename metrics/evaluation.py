import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch

def __log_metrics(logger, metrics, cm):
    # Overall metrics
    logger.info('')
    logger.info("--- Overall Performance --")
    logger.info("AUCROC: \t\t\t{:.4f}".format(metrics['AUCROC']))
    logger.info("Accuracy: \t\t\t{:.4f}".format(metrics['Accuracy']))
    logger.info("FPR: \t\t\t\t{:.4f}".format(metrics['FPR']))
    logger.info("TPR: \t\t\t\t{:.4f}".format(metrics['TPR']))
    logger.info("Precision: \t\t\t{:.4f}".format(metrics['Precision']))
    logger.info("F1-score: \t\t\t{:.4f}".format(metrics['F1-score']))
    logger.info("Threshold: \t\t\t{:.4f}".format(metrics['optimal_threshold']))
    logger.info('')

    # TPR per attack
    logger.info("------- TPR per Attack -------")
    for attack, tpr in metrics['tpr_per_attack'].items():
        attack = attack + ':\t\t\t' if len(attack) < 15 else attack + ':\t'
        logger.info(f"{attack}{tpr:.4f}")
    logger.info('')

    # Confusion matrix
    logger.info("------ Confusion Matrix ------")
    tn, fp, fn, tp = cm
    total = tn + fp + fn + tp
    tn_str = f'{tn} ({tn / (total) * 100:.2f}%)'
    fp_str = f'{fp} ({fp / (total) * 100:.2f}%)'
    fn_str = f'{fn} ({fn / (total) * 100:.2f}%)'
    tp_str = f'{tp} ({tp / (total) * 100:.2f}%)'
    logger.info("TN: {} \tFP: {}".format(tn_str, fp_str))
    logger.info("FN: {} \tTP: {}".format(fn_str, tp_str))
    logger.info('')

def __get_threshold_youden_index(y_true, y_pred):
    # If tensors are on GPU, move them to CPU and convert to NumPy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Calculate Youden index to determine optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_index]
    return optimal_threshold

def get_tpr_per_attack(y_labels, y_pred):
    aux_df = pd.DataFrame({'Label':y_labels,'prediction':y_pred})
    total_per_label = aux_df['Label'].value_counts().to_dict()
    correct_predictions_per_label = aux_df.query('Label != "Normal" and prediction == True').groupby('Label').size().to_dict()
    tpr_per_attack = {}
    for attack_label, total in total_per_label.items():
      if attack_label == 'Normal':
        continue
      tp = correct_predictions_per_label[attack_label] if attack_label in correct_predictions_per_label else 0
      tpr = tp/total
      tpr_per_attack[attack_label] = tpr
    return tpr_per_attack

def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    precision = tp/(tp+fp)
    f1 = (2*tpr*precision)/(tpr+precision)
    return {'Accuracy':acc,'TPR':tpr,'FPR':fpr,'Precision':precision,'F1-score':f1}


def compute_metrics(labels, y_pred, logger=None):
    # If tensors are on GPU, move them to CPU and convert to NumPy arrays
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = [0 if l == 'Normal' else 1 for l in labels]

    aucroc = roc_auc_score(y_true, y_pred)

    optimal_threshold = __get_threshold_youden_index(y_true, y_pred)
    result = get_overall_metrics(y_true, y_pred > optimal_threshold)

    tpr_per_attack = get_tpr_per_attack(labels, y_pred > optimal_threshold)

    overall_metrics = {'AUCROC': aucroc, **result, 'optimal_threshold': optimal_threshold}
    metrics_serializable = {k: float(v) for k, v in overall_metrics.items()}
    metrics_serializable['tpr_per_attack'] = tpr_per_attack

    if logger is not None:
        __log_metrics(logger, metrics_serializable, confusion_matrix(y_true, y_pred > optimal_threshold).ravel())

    return metrics_serializable
