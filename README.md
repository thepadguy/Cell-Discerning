# Cell-Discerning
Python program that identifies hematoxylin stained nuclei on a png image and uses them to create a Voronoi diagram.
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
of m*n dimensions, specified by the user, where each kernel entry is equal to 1/(distance to center of kernel).
A 3*3 distance kernel would be the following:
$$
\left(\begin{array}{cc}
0.70710678 & 1 & 0.70710678\\
1 & 0 & 1\\
0.70710678 & 1 & 0.70710678
\end{array}\right)
$$
This kernel convolutes an image where everything is black except what is inside the contours.
The resulting one-channel image is passed as the alpha channel to the original image, this way areas of high density are brighter
than areas of low density.

## How to use
Make sure that you have all the necessary python modules already installed and then simply follow the on-screen instructions.

## How it works
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

## Required python 3.x modules
* matplotlib (pyplot and mplPath)
* scikit-image
* scikit-learn
* scipy (spatial, stats, signal)
* numpy
* tqdm

## Development details
* By: thepadguy
* On: notepad++ & windows cmd
* Duration: 3 days
