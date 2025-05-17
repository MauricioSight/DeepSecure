import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch

def log_resource_metrics(logger, metrics):
    logger.info(f"--------- Resource Performance --------")
    logger.info(f"CPU Inference Time: \t{metrics['cpu_inference_time']:.6f} ms")
    logger.info(f"GPU Inference Time: \t{(metrics['gpu_inference_time'] or 0):.6f} ms")
    logger.info(f"MPS Inference Time: \t{(metrics['mps_inference_time'] or 0):.6f} ms")
    logger.info(f"Model size: \t\t{metrics['model_size_mb']:.6f} MB")
    logger.info('')

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

def pytorch_inference_time_cpu(model, input_data, n_reps=500):
    """Compute a pytorch model inference time in a cpu device"""

    model.to("cpu")
    input_data = input_data.to("cpu")

    starter, ender = 0, 0
    timings = np.zeros((n_reps, 1))

    with torch.no_grad():
        for rep in range(n_reps):
            starter = time.time()
            _ = model(input_data)
            ender = time.time()

            elapsed_time = ender - starter
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def pytorch_inference_time_gpu(model, input_data, n_reps=500, n_gpu_warmups=100):
    """Compute a pytorch model inference time in a gpu device"""
    # References:
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # https://discuss.pytorch.org/t/elapsed-time-units/29951 (time in milliseconds)

    model.to("cuda")
    input_data = input_data.to("cuda")

    # Init timer loggers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_reps, 1))

    # GPU Warm-up
    for _ in range(n_gpu_warmups):
        _ = model(input_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            _ = model(input_data)
            ender.record()
            # Wait for gpu to sync
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender)
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def pytorch_inference_time_mps(model, input_data, n_reps=500, n_gpu_warmups=100):
    """Compute a pytorch model inference time in a mps device"""
    # References:
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # https://discuss.pytorch.org/t/elapsed-time-units/29951 (time in milliseconds)

    model.to("mps")
    input_data = input_data.to("mps")

    # Init timer loggers
    starter, ender = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    timings = np.zeros((n_reps, 1))

    # GPU Warm-up
    for _ in range(n_gpu_warmups):
        _ = model(input_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            _ = model(input_data)
            ender.record()
            # Wait for gpu to sync
            torch.mps.synchronize()
            elapsed_time = starter.elapsed_time(ender)
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def get_inference_time(model, dummy_input):
    """Get the inference time of a model from all devices"""
    cpu_inference_time = None
    gpu_inference_time = None
    mps_inference_time = None

    # Check if the model is on GPU
    if torch.backends.mps.is_available():
        mps_inference_time = pytorch_inference_time_mps(model, dummy_input)
    
    if torch.cuda.is_available():
        gpu_inference_time = pytorch_inference_time_gpu(model, dummy_input)

    cpu_inference_time = pytorch_inference_time_cpu(model, dummy_input)

    return cpu_inference_time, gpu_inference_time, mps_inference_time

def pytorch_compute_model_size_mb(model):
    """Compute a pytorch model size in megabytes"""
    # Reference
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_buffer_mb = (param_size + buffer_size) / (1024) ** 2

    return size_param_buffer_mb