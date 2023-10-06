# Overview of Applied Data Science Coursework

## Visualisation
Visualisations made with explainability in mind. A priority of the project was to effectively communicate effects of a cyclone to the citizens of Bangladesh. 

https://user-images.githubusercontent.com/40215823/183653555-6cfdba17-5c40-409a-a2f3-d081105e3d86.mp4

I used examples from live emergency weather broadcasts to make this visualisation. I first used the cartopy package to load a projection surface map of the area into MatPlotLib. I then segmented the data into areas of high wind-speed and low wind speed, giving each a color to indicate severity. Finally I loaded country boarders and added cities names in Bengali to provide a more concrete grounding for the data. 

## Centroids

I devised an algorithm based on K-means and GMM to find the centre of a cyclone. A derivation and explanation can be found in [this Jupyter Notebook](../exploration/clustering.ipynb). The equation I derived is as follows:

$$
\huge
\mu_k = \frac{\sum_n f(\textbf{x}_n)^\alpha \textbf{x}_n}{\sum_n f(\textbf{x}_n)^\alpha}
$$

### Initial testing
<img src="centroid_threshold.gif" alt="Centroid" width="500"/>

### Full Visualisation
This is a depiction of the paths of all of the cyclones in our dataset

<img src="allpaths4p4.png" alt="All Paths" width="600"/>
