# Cell-Discerning
Python program that identifies hematoxylin stained nuclei on a png image and uses them to create a Voronoi diagram.

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
* scipy (spatial, stats, signal)
* numpy
* tqdm

## Development details
* By: thepadguy
* On: notepad++ & windows cmd
* Duration: 3 days
