import numpy as np


def extract_mosaic_relevance(heatmap):
    heatmap_width, heatmap_height = heatmap.shape
    quadrant_0 = heatmap[0:int(heatmap_width / 2), 0:int(heatmap_height / 2)]
    quadrant_1 = heatmap[0:int(heatmap_width / 2), int(heatmap_height / 2):heatmap_height]
    quadrant_2 = heatmap[int(heatmap_width / 2):heatmap_width, 0:int(heatmap_height / 2)]
    quadrant_3 = heatmap[int(heatmap_width / 2):heatmap_width, int(heatmap_height / 2):heatmap_height]
    relevance_per_quadrant = [quadrant_0, quadrant_1, quadrant_2, quadrant_3]
    return relevance_per_quadrant

def compute_metric(heatmap, target_category, order, metric='precision'):
    """ Computes pseudo-metrics for feature importance on images.

    Args:
        heatmap (np.ndarray): 2D-array with feature importance of an image
        target_category (int): index of target category of the heatmap
        order (list): list with the category indices of the original image for which the heatmap was created
        metric (str, optional): 'precision', 'sensitivity', 'false-negative-rate','false-positive-rate', 'specificity', 'accuracy','f', 'all'. Defaults to 'precision'.

    Raises:
        NotImplementedError: if chosen metric on feature importance is not implemented

    Returns:
        float/7-tuple: result for chosen metric. 7-tuple with all implemented metrics if metric == 'all'
    """
    if metric not in ['precision', 'sensitivity', 'false-negative-rate','false-positive-rate', 'specificity', 'accuracy','f', 'all']:
        raise NotImplementedError
    
    relevance_per_quadrant = extract_mosaic_relevance(heatmap)
    idx_true_target = np.where(np.array(order) == target_category)[0]
    idx_false_target = np.where(np.array(order) != target_category)[0]
    
    # true positive relevance -> positive feature importance on the images of the target class
    tp_relevance = np.sum([np.sum(relevance_per_quadrant[idx_true_target[i]][relevance_per_quadrant[idx_true_target[i]] > 0]) for i in range(len(idx_true_target))])
    # false negative relevance -> negative feature importance on the images of the target class
    fn_relevance = np.sum([np.sum(abs(relevance_per_quadrant[idx_true_target[i]][relevance_per_quadrant[idx_true_target[i]] < 0])) for i in range(len(idx_true_target))])
    # false positive relevance -> positive feature importance on the images of classes different from the target class
    fp_relevance = np.sum([np.sum(relevance_per_quadrant[idx_false_target[i]][relevance_per_quadrant[idx_false_target[i]] > 0]) for i in range(len(idx_false_target))])
    # true negative relevance -> negative feature importance on the images of classes different from the target class
    tn_relevance = np.sum([np.sum(abs(relevance_per_quadrant[idx_false_target[i]][relevance_per_quadrant[idx_false_target[i]] < 0])) for i in range(len(idx_false_target))])

    if metric == 'precision':
        if tp_relevance + fp_relevance == 0:
            print('No positive relevance encountered, probably something wrong with the heatmap generation. Precision set to None')
            return None
        return tp_relevance/(tp_relevance + fp_relevance)
    elif metric == 'sensitivity':
        return tp_relevance/(tp_relevance + fn_relevance)
    elif metric == 'false-negative-rate':
        return fn_relevance/(tp_relevance + fn_relevance)
    elif metric == 'false-positive-rate':
        return fp_relevance / (tn_relevance + fp_relevance)
    elif metric == 'specificity':
        return tn_relevance / (tn_relevance + fp_relevance)
    elif metric == 'accuracy':
        return (tp_relevance + tn_relevance) / (tp_relevance + tn_relevance + fn_relevance + fp_relevance)
    elif metric == 'f':
        precision = tp_relevance/(tp_relevance + fp_relevance)
        recall = tp_relevance/(tp_relevance + fn_relevance)
        return (2*precision*recall)/(precision + recall)
    elif metric == 'all':
        if tp_relevance + fp_relevance == 0:
            print('No positive relevance encountered, probably something wrong with the heatmap generation. Precision set to 0')
            precision = 0
        else:
            precision = tp_relevance/(tp_relevance + fp_relevance)
        sensitivity = tp_relevance/(tp_relevance + fn_relevance)
        recall = sensitivity
        fNrate = fn_relevance/(tp_relevance + fn_relevance)
        fPrate = fp_relevance / (tn_relevance + fp_relevance)
        specificity = tn_relevance / (tn_relevance + fp_relevance)
        accuracy = (tp_relevance + tn_relevance) / (tp_relevance + tn_relevance + fn_relevance + fp_relevance)
        if precision + recall == 0:
            f = 0
            print('No true positive relevance encountered, f-metric set to 0.')
        else:
            f = (2*precision*recall)/(precision + recall)
        return precision, sensitivity, fNrate, fPrate, specificity, accuracy, f
    else:
        raise NotImplementedError
