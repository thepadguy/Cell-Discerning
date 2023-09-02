# Cell-Discerning
Collection of Python scripts that analyse .png histological images. (Coded for use with H&E stains however you can modify to suit your needs).
### Contains
* Cell-Voronoi.py which creates a Voronoi plot with the centroids of the nuclei as input points.
* CellVoronoiClusters.py which does the same thing as above but uses the OPTICS algorithm to cluster the similar Voronoi regions
* CellNeighbours.py which can highlight regions of high cell density (more details can be found in the Update section)
* RelativeMorphometricAnalysis.py which is described in the "Update-RelativeMorphometricAnalysis.py" section

## Update
Now Cell-Discerning contains CellVoronoiClusters.py, which uses the sklearn OPTICS algorithm to cluster the voronoi
regions created (in CellVoronoiClusters) according to their area, eccentricity, perimeter, pearson correlation coefficient
(and p-value) and Shannon entropy (entropy is specifically calculated with a 0 mask on the rest of the image). In order
to calculate the minimum samples that define a cluster, a maximization routine of the silhouette measure is used from 2
sample to half the regions.
Note that for CellVoronoiClusters you may use "one contour subplot less" than for Cell-Voronoi due to sensitivity and
the features themselves.
Also, no running-images are uploaded for CellVoronoiClusters.py since they are pretty much the same.

The new update includes Cell-Neighbours.py, which finds contours like the above programs. Then it creates a distance kernel
of m*n dimensions, specified by the user, where each kernel entry is equal to 1 / (distance to center of kernel).
A 3x3 distance kernel would be the following:
\
[[0.70710678, 1, 0.70710678], [1,0,1], [0.70710678, 1, 0.70710678]]\
This kernel convolutes an image where everything is black except what is inside the contours.
The resulting one-channel image is passed as the alpha channel to the original image, this way areas of high density are brighter
than areas of low density.

## Update - RelativeMorphometricAnalysis.py
This is the biggest update as of now, hence it has its own update section on the README file.
RelativeMorphometricAnalysis.py (or RMA.py) is a Python program that does "Morphometric Analysis of Tissue Distribution of Mobile Cells
in Relation to Immobile Tissue Structures". In other words, it implements the procedure described in the following paper:
\
Nikitina, L., Ahammer, H., Blaschitz, A., Gismondi, A., Glasner, A., Schimerk, M.  G., Dohr, G., & Sedlmayr, P. (2011). A new method for morphometric analysis of tissue distribution of mobile cells in relation to immobile tissue structures. PLoS ONE, 6(3). https://doi.org/10.1371/journal.pone.0015086
\
albeit with some differences from the exact procedure described. Specifically, the random pixel incrementation is done by randomly choosing one of the hard-coded expansion kernels and applying it to the image until all "cells" reach the average area of all the cells detected in the samples.
As for the cell detection, in the paper it is done by thresholding the RGB values according to how
the cells were marked in the tissue sampes; whereas in RMA.py it is done by separating a hematoxylin stain layer and thresholding it in grayscale (by applying the Yen threshold) in order to locate the nuclei (so it actually is a nucleus identification, however the program can be modified to locate cells by RGB thresholding if provided with a good resolution image). Also, since it locates nuclei this way it implements a noise filtering routine like the one described in the "How it works - CellVoronoi.py" section.\
RMA.py can be used either with one sample or with more than one samples.\
If used with one sample, then its fN & fR values are normalized based on the normalization of the random (simulated) values (min-max normalization in 0..1 range) and they are plotted as a point in the quantile-quantile plot of the random values. The distances of all points (including the sample's point) from the 1st degree polynomial regression line are displayed in a boxplot besides the Q-Q plot with the sample's distance displayed as a red point.\
If used with more than one samples (provided they don't all give the same fN & fR values), the end result is again a Q-Q plot along with two boxplots. For the Q-Q plot, the fN&fR values of random simulations and of the samples are normalized (min-max normalization in 0..1 range) separately. The two boxplots are boxplots of the distances of the sample points from the 1st deg polynomial regression line (one plot for the samples and one plot for the random simulations).
### Required modules for RMA.py
* matplotlib (pyplot and path)
* skimage
* scipy (stats and signal)
* numpy
* numpy.ma
* tqdm
* random (comes with python)
* seaborn

## How to use
Make sure that you have all the necessary python modules already installed and then simply follow the on-screen instructions.

## How it works - CellVoronoi.py
The program first reads a png image, and in case it has more than 3 channels, keeps only 3 channels (deletes alpha). Then it,
separates the hematoxylin stain, turns it to grayscale and uses the Yen threshold filter on it. After that, it finds the
contour lines on the image, which now should be the nuclei and small noise that aren't nuclei. The program assumes that the
set of the areas of all nuclei has a skewness of 0 and that the set of the areas of all "noise-particles" also has a skewness
of 0. So it generates many cutoff values to separate those two sets, and tries to find those sets that have as small skewness
as possible (A->0, B->0, => A+B -> 0, A and B > 0). However, there might be contours that encircle noise-shapes with area bigger than the average nucleus area.
To solve this problem the program generates a list of possible separations with minimal skewness values and lets the user choose
which separation to keep. After that it searches the kept contours to find contours that are inside other contours and delete them
since it regards them as noise. Then, it finds the centroid of each contour that has made till here and uses those centroids
to generate a Voronoi diagram which it superimposes on top of the input image.

## Required python 3.x modules (please look at the files you want to use as the required modules may differ)
* matplotlib (pyplot and mplPath)
* scikit-image
* scikit-learn
* scipy (spatial, stats, signal)
* numpy
* tqdm

## Development details
* By: thepadguy
* On: notepad++ & windows cmd
* Duration: 17 days
