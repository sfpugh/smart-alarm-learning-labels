import numpy as np

ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

def threshold_predict(p, threshold):
    """
    Threshold the prediction given the probabilities
    
    Parameters:
        p - array of probabilities for each label 
        threshold - threshold of probability to make prediction
    Returns:
        int - prediction of supprressible or not suppressible
    """
    if p[SUPPRESSIBLE] > p[NOT_SUPPRESSIBLE] and p[SUPPRESSIBLE] > threshold:
        return SUPPRESSIBLE
    elif p[NOT_SUPPRESSIBLE] > p[SUPPRESSIBLE] and p[NOT_SUPPRESSIBLE] > threshold:
        return NOT_SUPPRESSIBLE
    else:
        return ABSTAIN

def threshold_suppressible_predict(p, threshold):
    """
    Threshold the prediction of suppressible alarms, i.e., in order to
    predict suppressible the P(SUPPRESSIBLE | LFs) must be the majority
    and greater than the threshold
    
    Parameters:
        p - array of probabilities for each label 
        threshold - threshold of probability to make prediction
    Returns:
        int - prediction of supprressible or not suppressible
    """
    if p[SUPPRESSIBLE] > p[NOT_SUPPRESSIBLE] and p[SUPPRESSIBLE] > threshold:
        return SUPPRESSIBLE
    elif p[NOT_SUPPRESSIBLE] > p[SUPPRESSIBLE]:
        return NOT_SUPPRESSIBLE
    else:
        return ABSTAIN

def predict_at_abstain_rate(Y_prob, abstain_target, t_lower=0.5, t_upper=1.0):
    """
    Parameters:
        Y_pred - matrix of probabilities of each label per instance
        abstain_target - abstain rate to achieve with predictions

    Returns:
        numpy int array - predictions of suppressible or not suppressible
    """

    for threshold in np.arange(0.5,1.0,0.1):
        Y_pred = np.apply_along_axis(threshold_predict, 1, Y_prob, t_mid)
        abstain_rate = np.sum(Y_pred == ABSTAIN) / len(Y_pred)

        if abs(abstain_rate - abstain_target) < 0.05:
            return Y_pred