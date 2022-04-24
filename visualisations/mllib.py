import numpy as np

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