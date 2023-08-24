import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import skimage as ski
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import skew
from scipy.signal import find_peaks
import numpy as np
from tqdm import tqdm

print("[+] Imported modules.")

#finds contours on image and plots voronoi diagram on top of it
#there are too many points though

def centroid_of_contour(contour):
    #contour is in y,x we need to swap it to x,y
    vertices = contour.copy()   #to keep original array unchanged
    vertices[:, [0,1]] = vertices[:, [1,0]]
    x,y = 0,0
    n = len(vertices)
    signed_area = 0
    for i in range(len(vertices)):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i+1)%n]
        #shoehorse method
        area = (x0*y1) - (x1*y0)
        signed_area += area
        x += (x0+x1)*area
        y += (y0+y1)*area
    signed_area *= 0.5
    x /= 6*signed_area
    y /= 6*signed_area
    return x,y

def separate_histological_stains(image):
    """returns three images, from coloured image input,
    one with hematoxylin stain (nuclei), one with eosin (cytoplasm)
    and one with dab"""
    image_hed = ski.color.rgb2hed(image)
    null = np.zeros_like(image_hed[:, :, 0])
    image_h = ski.color.hed2rgb(np.stack((image_hed[:,:,0],null,null), axis=-1))
    image_e = ski.color.hed2rgb(np.stack((null, image_hed[:,:,1],null), axis=-1))
    image_d = ski.color.hed2rgb(np.stack((null, null, image_hed[:,:,2]), axis=-1))
    
    return image_h, image_e, image_d

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def pick_subplot(event):
    global chosen_subplot
    global index
    if event.button == 3:   #right mouse click
        chosen_subplot = index

def switch_subplot(event):
    global index
    if event.key == 'right':
        if index == (len(f_c_s)-1):
            index = 0
        else:
            index += 1
    elif event.key == 'left':
        if index == 0:
            index = (len(f_c_s)-1)
        else:
            index -= 1
    plt.clf()
    print("[++] Loading subplot no. {index}".format(index=index+1))
    plt.imshow(gray_grans, cmap=plt.cm.gray)  #less processing
    plt.axis('off')
    fig.canvas.manager.set_window_title("Contour subplot no. {i}".format(i=index+1))
    for j in f_c_s[index]:
        plt.plot(j[:,1], j[:,0], color='red')
    plt.draw()

print("[WELCOME] to Cell-Voronoi.py\nPlease use only png images and follow instructions.")
path = input("[INPUT] image path and press Enter: ")

grans = ski.io.imread(path)
print("[+] Read image.")
channels = grans.shape[2]
if channels > 3:
    print("[-] Image has {c} channels, reducing them to 3".format(c=channels))
    grans = grans[:,:,:3]
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("3-channel image")
    ski.io.imshow(grans); ax.axis('off'); plt.show()
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Original image")
ax.imshow(grans); ax.axis('off'); plt.show()
#separate hematoxylin (nucleus) stain
grans_h, grans_e, grans_d = separate_histological_stains(grans)
print("[+] Separated hematoxylin stains.")
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Hematoxylin stain")
ax.imshow(grans_h); ax.axis('off'); plt.show()
#to grayscale
gray_grans = ski.color.rgb2gray(grans_h)
print("[+] Image to grayscale, done")
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Hematoxylin stain - grayscale")
ax.imshow(gray_grans, cmap=plt.cm.gray); ax.axis('off'); plt.show()
#threshold for black nuclei
threshold = ski.filters.threshold_yen(gray_grans)
threshed_gray_grans = gray_grans > threshold
print("[+] Thresholded image.")
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Yen threshold (over) on h_stain_grayscale")
ax.imshow(threshed_gray_grans, cmap=plt.cm.gray); ax.axis('off'); plt.show()
#find contours on thresholded image
contours = ski.measure.find_contours(threshed_gray_grans)
print("[+] Found contours, visualizing them.")
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Contours on original image")
ax.imshow(grans); ax.axis('off')
for i in contours:
    ax.plot(i[:,1], i[:,0], color='red')
plt.show()
#there might be a lot of random noise, so we assume that there are
#two sets of contours, those of nuclei, and those of small dots
#and that both sets are uniform distributions (skew=0), so we need to
#split the areas into those two specific datasets, and calculate the skewness
#of those two each time, then find min(skew1+skew2) since then they are both
#tending to 0.
#we use a range of 1000 to be as accurate as possible while keeping a
#computationally reasonable time
#However, there might be nan values in the "under" list, if a list has len<=1
#we stop the explorative splitting and find the minimum sum.
print("[+] Filtering noise based on area.")
areas = []
for i in contours:
    areas.append(PolyArea(i[:,1], i[:,0]))
areas = np.array(areas)
skewenesses_sum = []
st = 0
for i in tqdm(range(1,1001)):
    cutoff = areas.max() - i*((areas.max()-areas.min())/1000)
    areas1 = [j for j in areas if j >= cutoff]
    areas2 = [j for j in areas if j < cutoff]
    if (len(areas1) <= 1) or (len(areas2) <= 1):
        skewenesses_sum.append(np.nan)
    else:
        skew1 = skew(areas1)
        skew2 = skew(areas2)
        skewenesses_sum.append(np.abs(skew1+skew2))
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Skewness minimization sum")
ax.plot(skewenesses_sum, color='blue');
plt.title("Skewness minimization sum"); plt.show()

#find local minima of skewness minimization
#first purge array of nan
skewenesses_sum_valid = np.array([(-1)*i for i in skewenesses_sum if not(np.isnan(i))])
#print(skewenesses_sum_valid)
padding_left = 0
for i in range(len(skewenesses_sum)):
    if not np.isnan(skewenesses_sum[i]):
        break
    padding_left += 1
local_minima_index = find_peaks(skewenesses_sum_valid)[0]
if len(local_minima_index) != 0:    #could not find any peaks
    for idx, value in enumerate(local_minima_index):
        local_minima_index[idx] = value + padding_left + 1 #due to 0 indexing
elif len(local_minima_index) == 0:  #to cover edge case
    print("[-] Could not find peaks, using NumPy minimum identification instead, only 1 cutoff.")
    minimum = skewenesses_sum.index(np.nanmin(skewenesses_sum))
    local_minima_index = [minimum]
cutoffs = [areas.max() - i*((areas.max()-areas.min())/1000) for i in local_minima_index]
print("[i] COEFFICIENTS ARE: "+str(local_minima_index))
print("[i] CUTOFFS ARE: "+str(cutoffs))
f_c_s = []
for cutoff in cutoffs:
    f_c = []
    for i in range(len(areas)):
        if areas[i] >= cutoff:
            f_c.append(contours[i])
    f_c_s.append(f_c)
fig, ax = plt.subplots()
if len(cutoffs) > 1:
    #user message
    print("[!] {i} best possible contour-segmentation-modes (nuclei-identification-modes) have been found. Use left and right arrow keys to navigate through them. Right click on the one you want to use. Then close the window.".format(i=len(f_c_s)))
    index = 0
    chosen_subplot = 0
    ax.imshow(gray_grans, cmap=plt.cm.gray)
    ax.axis('off')
    for j in f_c_s[0]:
        ax.plot(j[:,1], j[:,0], color='red')
    cid_key = fig.canvas.mpl_connect("key_press_event", switch_subplot)
    cid_choose = fig.canvas.mpl_connect("button_press_event", pick_subplot)
    fig.canvas.manager.set_window_title("Contours subplot no. {i}".format(i=index+1))
    fig.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid_key)
    fig.canvas.mpl_disconnect(cid_choose)
    print("[+] Your final chosen subplot is subplot no. {i}.".format(i=chosen_subplot+1))#0indexing
    filtered_contours = f_c_s[chosen_subplot]
else:   #ax is not subscriptable if it has length of 1
    print("[!] 1 best possible contour-segmentation-mode (nuclei-identification-mode) has been found and will automatically be used. Close the window to continue.")
    ax.imshow(grans)
    for j in f_c_s[0]:
        ax.plot(j[:, 1], j[:, 0], color='red')
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    chosen_subplot = 0
    filtered_contours = f_c_s[chosen_subplot]

#some contours may contain other contours, let's use a crude way to filter them
non_nested_contours = []
con1 = filtered_contours.copy()  #don't alter original contours
poly_paths = []
for i in tqdm(con1):
    poly_paths.append(mplPath.Path(i))
    #note here that a path has (x,y) but contours have(y,x)
    #however here it doesn't matter because we need to filter
    #nested contours and since they are all transformed, we don't care
non_nested_paths = []
should_I_run_again = True
generations_count = 0
while should_I_run_again:
    should_I_run_again = False
    for parent in tqdm(poly_paths):
        for j in poly_paths:
            if j == parent:
                continue
            j_index_poly = poly_paths.index(j)
            if j in non_nested_paths:
                if parent.intersects_path(j):
                    should_I_run_again = True
                    j_index_nest = non_nested_paths.index(j)
                    del(non_nested_paths[j_index_nest])
                    del(poly_paths[j_index_poly])
                    del(con1[j_index_poly])
            else:
                non_nested_paths.append(j)
                if parent.intersects_path(j):
                    should_I_run_again = True
                    del(poly_paths[j_index_poly])
                    del(con1[j_index_poly])
                    non_nested_paths.pop()
    generations_count += 1
    print("[i] Runs: {r}".format(r=generations_count))
print("[+] Filtered nested contours, now there are {length} contours.".format(length=len(non_nested_paths)))
for i in range(len(non_nested_paths)):
    non_nested_contours.append(con1[i])
print("[+] Visualizing contours.")
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Filtered nested contours")
ax.imshow(threshed_gray_grans, cmap=plt.cm.gray)
for contour in non_nested_contours:
    ax.plot(contour[:,1], contour[:,0], color='red')
ax.axis('off'); ax.set_xticks([]); ax.set_yticks([]); plt.show()


#find centroids of each contour area
centroids = []
for i in tqdm(non_nested_contours):
    x,y = centroid_of_contour(i)
    centroids.append([x,y])
print("[+] Calculated centroids.")
#plot voronoi along image
print("[+] Plotting voronoi diagram.")
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True)
fig.canvas.manager.set_window_title("Nucleui voronoi diagram")
ax1.imshow(grans)
ax1.axis("off")
ax2.imshow(grans)
vor = Voronoi(centroids)
voronoi_plot_2d(vor, ax=ax2, show_points=False, show_vertices=False, line_colors='red')
ax2.invert_yaxis()
ax2.axis("off")
fig.tight_layout()
plt.show()
