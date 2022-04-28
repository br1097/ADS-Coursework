import numpy as np
import scipy.stats as stats

def mag_m_step(frame: np.array, lng: np.array, lat: np.array):
    assert frame.shape == (len(lat), len(lng))
    # The sum of responsibilities in the denominator
    sum_resp = 0
    # weighted points, the numerator
    xs = [0, 0]
    # The order put on the responsibility
    alpha = 4
    
    for lt, row in zip(lat, frame):
        for lg, val in zip(lng, row):
            inc = val ** alpha
            xs[0] += lt * inc
            xs[1] += lg * inc
            
            sum_resp += inc
            
    return np.divide(xs, sum_resp)

def z_score_threshold(variable):
    np.array(
            [[ 
                ] 
                ])
    above = np.full(variable.shape, np.nan)
    below = np.full(variable.shape, np.nan)
    for ens_idx, (ensemble, ensemble_score) in enumerate(zip(variable, stats.zscore(variable, axis=None))):
        for fr_idx, (frame, frame_score) in enumerate(zip(ensemble, ensemble_score)):
            above[ens_idx][fr_idx] = np.where(frame_score >= 3, frame, np.nan)
            below[ens_idx][fr_idx] = np.where(frame_score < 3, frame, np.nan)
    
    return above, below
    