import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def mag_m_step(frame: np.array, long: np.array, lat: np.array):
    # The sum of responsibilities in the denominator
    sum_resp = 0
    # weighted points, the numerator
    xs = [0, 0]
    # The order put on the responsibility
    alpha = 4
    
    for x in range(len(lat)):
        for y in range(len(long)):
            if x < frame.shape[0] and y < frame.shape[1]:
                fn = frame[x][y]
                xs += np.multiply([long[y], lat[x]], fn**alpha)
                
                sum_resp += fn**alpha
            
    return np.divide(xs, sum_resp)

def visualise(output_filename: str, long: np.ndarray, lat: np.ndarray, variable: np.ndarray, centroids=None):
    """
    just a bog standard visualisation function for putting a time series variable on a long vs lat graph
    variable should be a time series of frames
    """
    filenames = []
    maxlevel = variable.max()
    minlevel = variable.min()
    steps = 7
    levels = [((maxlevel - minlevel)/steps) * val for val in list(range(steps))]
    
    for idx, frame in enumerate(variable):
        fig, ax = plt.subplots()
        cntr = ax.contourf(long, lat, frame, levels=levels)
        fig.colorbar(cntr, ax=ax)
        if centroids:
            ax.plot(centroids[idx][0], centroids[idx][1], "ro")
        filename = f"vis/sc{idx}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()
        
    # build gif
    with imageio.get_writer(f"vis/{output_filename}", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)