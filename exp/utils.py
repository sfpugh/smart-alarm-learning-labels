import sys
import logging
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
    Threshold the prediction of suppressible alarms, i.e., in order to predict suppressible the 
    P(SUPPRESSIBLE | LFs) must be the majority and greater than the threshold
    
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


def predict_at_abstain_rate(Y_prob, abstain_target):
    """
    Use Binary Search to find a threshold at which the abstain rate is close to the target

    Parameters:
        Y_prob - matrix of probabilities of each label per instance
        abstain_target - abstain rate to achieve with predictions
    Returns:
        numpy int array - predictions of suppressible or not suppressible
    """
    t_lo, t_hi = 0.5, 1.0
    tol = 0.005

    # Check if base abstain rate is too high such that the target rate cannot be obtained, and
    # if the base abstain rate achieves the target
    Y_pred = np.apply_along_axis(threshold_predict, 1, Y_prob, t_lo)
    abstain = np.sum(Y_pred == ABSTAIN) / len(Y_pred)
    if abstain > abstain_target + tol:
        raise Exception("Base abstain of {:.3f} rate is too high.".format(abstain))
    elif (abstain >= abstain_target - tol) and (abstain < abstain_target + tol):
        return Y_pred

    Y_pred = np.apply_along_axis(threshold_predict, 1, Y_prob, t_hi)
    abstain = np.sum(Y_pred == ABSTAIN) / len(Y_pred)
    if (abstain >= abstain_target - tol) and (abstain < abstain_target + tol):
        return Y_pred

    # Otherwise BinarySearch
    for n in range(50):
        t_mid = t_lo + (t_hi - t_lo) / 2
        Y_pred = np.apply_along_axis(threshold_predict, 1, Y_prob, t_mid)
        abstain = np.sum(Y_pred == ABSTAIN) / len(Y_pred)
        #print("A {:.18f}, L {}, H {}".format(abstain,t_lo,t_hi))
        
        if abstain > abstain_target:
            t_hi = t_mid
        else:
            t_lo = t_mid

    # If BinarySearch was successful then we are done; otherwise, toss biased coin
    Y_pred_lo = np.apply_along_axis(threshold_predict, 1, Y_prob, t_lo)
    abstain_lo = np.sum(Y_pred_lo == ABSTAIN) / len(Y_pred_lo)
    if (abstain_lo >= abstain_target - tol) and (abstain_lo < abstain_target + tol):
        return Y_pred_lo

    Y_pred_hi = np.apply_along_axis(threshold_predict, 1, Y_prob, t_hi)
    abstain_hi = np.sum(Y_pred_hi == ABSTAIN) / len(Y_pred_hi)
    if (abstain_hi >= abstain_target - tol) and (abstain_hi < abstain_target + tol):
        return Y_pred_hi

    # Toss coin
    idx_uncertain = np.argwhere( np.apply_along_axis(lambda Y: Y[0] == Y[1], 1, Y_prob) )
    p = (abstain_target - abstain_lo) / (abstain_hi - abstain_lo)

    for i in range(25):
        Y_pred_hi[idx_uncertain] = np.random.choice([ABSTAIN, NOT_SUPPRESSIBLE, SUPPRESSIBLE], size=(len(idx_uncertain),1), p=[p, (1-p)/2, (1-p)/2])
        abstain = np.sum(Y_pred_hi == ABSTAIN) / len(Y_pred_hi)
        print("@ p ", abstain)
        if (abstain >= abstain_target - tol) and (abstain < abstain_target + tol):
            return Y_pred_hi 
    
    raise Exception("Could not achieve target abstain rate.")