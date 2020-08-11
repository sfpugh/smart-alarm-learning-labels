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