import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import skimage as ski
from scipy.stats import skew
from scipy.signal import find_peaks, convolve2d
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import random
import seaborn as sns

print("[+] Imported modules.")

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
    plt.imshow(At_for_view)  #currently is originalimage * mask_keep
    plt.axis('off')
    fig.canvas.manager.set_window_title("Contour subplot no. {i}".format(i=index+1))
    for j in f_c_s[index]:
        plt.plot(j[:,1], j[:,0], color='red')
    plt.draw()

def create_dilation_kernel(height, width):
    arr = np.zeros_like(np.arange(height*width).reshape((height,width)),dtype=np.uint8)
    arr.fill(1)
    return arr

def distance_of_point_from_line(slope, intercept, x,y):
    return np.abs(slope*x-y+intercept)/np.sqrt(slope**2 + 1)

print("[WELCOME] to RMA.py")
print("[IMPORTANT INFO] RMA.py implements the techniques described in the following paper, albeit with some modifications. \
For example, the random incrementation of pixels in the simulated images is done by incrementing all pixels with the same \
randomly chosen expansion kernel (expansion kernels are hardcoded so you can see them in this file), also the \
cells selection isn't done based on their relative RGB values like it is in the paper, however \
Hematoxylin-stained nuclei identification routine is implemented along with some noise filtering. (Of course, you may change the identification \
routine, but always specify that it is not the original. RMA.py can also be used with a single sample, attempting to do the procedure \
described in the paper but suiting it to work with a single sample.")
print("[IMPORTANT INFO] Citation of forementioned paper:")
print("\tNikitina, L., Ahammer, H., Blaschitz, A., Gismondi, A., Glasner, A., Schimek, M. G., Dohr, G., & Sedlmayr, P. (2011). A new method \
for morphometric analysis of tissue distribution of mobile cells in relation to immobile tissue structures. PLoS ONE, 6(3). \
https://doi.org/10.1371/journal.pone.0015086 ")
print("[IMPORTANT INFO] RMA.py is stored online in GitHub in the following repository:\
\n\t https://github.com/thepadguy/Cell-Discerning \n\
So if you decide to use results generated from the current python program, please cite both the above paper and the above GitHub repository.")
fN_list = []
fR_list = []
average_area_of_all_cells_list = []
first_image_width = 0; first_image_height = 0;

counter = 0
sample_flag = True
while sample_flag:
    sample_flag = False

    path = input("[INPUT] image path and press Enter: ")

    grans = ski.io.imread(path)
    print("[+] Read image.")
    if counter == 0:
        first_image_height = grans.shape[0]
        first_image_width = grans.shape[1]
    channels = grans.shape[2]
    if channels > 3:
        print("[-] Image has {c} channels, reducing them to 3".format(c=channels))
        grans = grans[:,:,:3]
        fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("3-channel image")
        ski.io.imshow(grans); ax.axis('off'); plt.show()
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Original image")
    ax.imshow(grans); ax.axis('off'); plt.show()
    #we need to define At now and from now on ignore everything outside of it
    #the mask should cover the background, lumen of vessels and everything that
    #would skew the analysis, e.g. glands
    print("[+] Using an image editing software, color green RGB(0,255,0) all the areas you want to ignore (because it might skew the analysis/detection) on the image and black RGB(0,0,0) everything you want to keep.")
    mask_path = input("[INPUT] mask's path and press Enter: ")
    mask = ski.io.imread(mask_path)
    mask = mask[:,:,:3] #in case it has more channels
    #skimage reads images as numpy.uint8 datatype, 0..255
    mask_g = mask[:,:,1]    #discard
    mask_throw = (mask_g/255).astype('int') #for usage with ma
    mask_keep = ~(mask_g/255).astype('int').astype('bool')  #turn to TrueFalse
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Mask - At")
    ax.imshow(mask_keep, cmap=plt.cm.gray); plt.show()
    #apply mask to image
    print("[+] Applying mask.") #min = 0, max = 255
    #At_for_view turns pixels to black, thus it changes the histogram.
    At_for_view = np.stack((grans[:,:,0]*mask_keep, grans[:,:,1]*mask_keep, grans[:,:,2]*mask_keep), axis=-1)
    fig, ax = plt.subplots(1,2); fig.canvas.manager.set_window_title("Original and kept image")
    ax[0].set_title("Original image"); ax[0].axis('off')
    ax[1].set_title("Image to keep"); ax[1].axis('off')
    ax[0].imshow(grans); ax[1].imshow(At_for_view)
    plt.show()

    #separating hematoxylin stain
    grans_h, grans_e, grans_d = separate_histological_stains(grans)
    print("[+] Separated hematoxylin stains.")
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Hematoxylin stain")
    At_for_view = np.stack((grans_h[:,:,0]*mask_keep, grans_h[:,:,1]*mask_keep, grans_h[:,:,2]*mask_keep), axis=-1)
    ax.imshow(At_for_view); ax.axis('off'); plt.show()
    #to grayscale
    gray_grans = ski.color.rgb2gray(grans_h)
    print("[+] Image to grayscale, done")
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Hematoxylin stain - grayscale")
    At_for_view = gray_grans*mask_keep
    ax.imshow(At_for_view, cmap=plt.cm.gray); ax.axis('off'); plt.show()

    #threshold for black nuclei (nuclei are under the threshold, so if > they are black)
    At = ma.array(grans, mask=np.stack((mask_throw, mask_throw, mask_throw), axis=-1))
    valids = At[~At.mask]
    threshold = ski.filters.threshold_yen(gray_grans)
    threshed_gray_grans = gray_grans <= threshold
    At_for_view = ~(threshed_gray_grans*mask_keep) #nuclei are black now, TrueFalse image, only black and white
    print("[+] Thresholded image.")
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Yen threshold (over) on h_stain_grayscale")
    ax.imshow(At_for_view, cmap=plt.cm.gray); ax.axis('off'); plt.show()

    #find contours on thresholded_image
    contours = ski.measure.find_contours(At_for_view)   #currently it is threshed_gray_grans*mask_keep
    print("[+] Found contours, visualizing them.")
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Contours on At")
    At_for_view = np.stack((grans[:,:,0]*mask_keep, grans[:,:,1]*mask_keep, grans[:,:,2]*mask_keep), axis=-1)
    ax.imshow(At_for_view); ax.axis('off')
    for i in contours:
        ax.plot(i[:,1], i[:,0], color='red')
    plt.show()

    #same as Cell-Voronoi, filter out noised - contours
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
        ax.imshow(At_for_view, cmap=plt.cm.gray)    #currently is originalimage*mask_keep
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
        ax.imshow(At_for_view)  #currently is originalimage*mask_keep
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
    ax.imshow(At_for_view, cmap=plt.cm.gray)    #currently is original image * mask_keep
    for contour in non_nested_contours:
        ax.plot(contour[:,1], contour[:,0], color='red')
    ax.axis('off'); ax.set_xticks([]); ax.set_yticks([]); plt.show()

    #below chapter is from CellNeighbours so that it creates an image with inside of contour = 1 and outside = 0
    print("[+] Creating Ac mask.")  #Ac = Area cells
    masked_image = np.zeros_like(gray_grans, dtype='uint8')
    for contour in tqdm(non_nested_contours):
        contour = np.array([np.round(contour[:,0]).astype('int'), np.round(contour[:,1]).astype('int')])
        contour = contour.T
        rr,cc = ski.draw.polygon(contour[:,0], contour[:,1], masked_image.shape)
        masked_image[rr, cc] = 1
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Masked-out contour polygons - Ac")
    ax.imshow(masked_image, cmap=plt.cm.gray)
    plt.show()

    #Now we need to do the same with fixed tissue structures
    #because the first mask might also include the background, we need
    #another mask with ft-structures (it might be the same but it might be different)
    #so the below mask keeps only fts, since none of them are in the background
    #it automatically filters over background
    print("[+] The same way as before, create a green RGB(0,255,0) mask over fixed-tissue structures (e.g. endothelium/muscle layers of vessels).")
    print("[NOTE] That the structures you will highlight now might also be included in the previous mask.")
    print("For example, you may already have greened-out an artery with its lining because it would tamper the analysis' results, now you would want to green-out only the artery's lining, since the neighbourhood calculation will be based on this lining.")
    ft_mask_path = input("[INPUT] fixed-tissue-structures' mask path and press Enter: ")
    ft_mask = ski.io.imread(ft_mask_path)
    ft_mask = ft_mask[:,:,:3]   #in case it has more channels
    #skimage reads images as numpy.uint8 datatype, 0..255
    ft_mask_g = ft_mask[:,:,1]  #fts are 255
    ft_mask_throw = (ft_mask_g/255).astype('int')   #for usage with ma, fts = 1 = throw
    ft_mask_keep = (ft_mask_g/255).astype('int').astype('bool')    #fts are True are keep
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Mask - Af")
    ax.imshow(ft_mask_keep, cmap=plt.cm.gray); plt.show()
    print("[+] Applying fts-mask.")
    Af_for_view = np.stack((grans[:,:,0]*ft_mask_keep, grans[:,:,1]*ft_mask_keep, grans[:,:,2]*ft_mask_keep), axis=-1)
    fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Fixed tissue structures")
    ax.imshow(Af_for_view); plt.show()

    #now time to calibrate the dilation kernel
    #using the while loop from cellNeighbours.py
    flag = True
    while flag:
        flag = False
        k_width = input("[INPUT] dilation kernel's width and press Enter: ")
        k_height = input("[INPUT] dilation kernel's height and press Enter: ")
        k_width = int(float(k_width)); k_height = int(float(k_height))
        kernel = create_dilation_kernel(k_height, k_width)
        
        Abn = ski.morphology.binary_dilation(ft_mask_keep, kernel).astype('int')
        An = Abn - ft_mask_keep.astype(Abn.dtype)
        An = An * mask_keep #of course do not violate At
        
        #currently At is originalimage * mask_keep, so 3 channels int
        Ar = ~Abn.astype('bool')    #True keep False throw
        #normally we should also do Ar = Ar*mask_keep, however
        #everything about Ar is calculated indirectly through Abn
        Ar_for_view = np.stack(((At_for_view[:,:,0]*(~Abn.astype('bool'))), (At_for_view[:,:,1]*(~Abn.astype('bool'))), (At_for_view[:,:,2]*(~Abn.astype('bool')))), axis=-1)
        
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.001, top=1, bottom=0)
        fig.canvas.manager.set_window_title("Fixed tissue structures dilation with {h}x{w} (hxw) kernel".format(h=k_height, w=k_width))
        ax0 = plt.subplot2grid((2,4), (0,0))
        ax0.imshow(ft_mask_keep, cmap=plt.cm.gray)
        ax0.set_title("FTS")
        ax1 = plt.subplot2grid((2,4), (0,1))
        ax1.imshow(Abn, cmap=plt.cm.gray)
        ax1.set_title("FTS w/ neighbourhood")
        ax2 = plt.subplot2grid((2,4), (0,2))
        ax2.imshow(An, cmap=plt.cm.gray)
        ax2.set_title("FTS neighbourhood")
        
        ax10 = plt.subplot2grid((2,4), (1,0))
        ax10.imshow(Af_for_view)
        #note that in ax[1][1] we do not also multiply by At for depiction reasons,
        #we prefer to show everything
        #HOWEVER if you need this image for computations you need to also multiply by
        #mask_keep every channel, because it removes background and uncounted for areas
        ax11 = plt.subplot2grid((2,4), (1,1))
        ax11.imshow(np.stack((grans[:,:,0]*Abn, grans[:,:,1]*Abn, grans[:,:,2]*Abn), axis=-1))
        ax12 = plt.subplot2grid((2,4), (1,2))
        ax12.imshow(np.stack((grans[:,:,0]*An, grans[:,:,1]*An, grans[:,:,2]*An), axis=-1))
        
        ax3 = plt.subplot2grid((2,4), (0,3), rowspan=2)
        ax3.imshow(Ar_for_view)
        ax3.set_title("Rest area")

        plt.show()
        
        confirm = input("[INPUT] If you want to try with different kernel dimensions, type 'y': ")
        if confirm.strip() == 'y':
            flag = True


    #now for finding the contours in the neighbourhoods and in the rest area
    #we could either take the contours and split them, or find them anew
    #finding them anew could be problematic due to noise concnetration in
    #either region, so it is better to split them.
    #However, some contours might be half on one region, and half on the other.
    #Supposing that this symbolizes cell area percentage, we will simply count
    #the fraction of the counter on either side.
    print("[+] Splitting contours on fts-neighbourhood region and rest area region.")
    #contours are stored in non_nested_contours array
    #also note that masked_image displays these contours as white (1) and everything else as black (0)
    Anc = An * masked_image
    #here, masked_image contains the cells as contours,
    #the cells are already inside At, so *Ar won't do anything bad
    #so it is unnecessary to do Ar *= mask_keep
    Arc = Ar * masked_image
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    fig.tight_layout()

    fig.canvas.manager.set_window_title("Separated cells")
    ax[0][0].imshow(Anc, cmap=plt.cm.gray)
    ax[0][0].set_title("Cells in fts-neighbourhood")
    ax[0][1].imshow(Arc, cmap=plt.cm.gray)
    ax[0][1].set_title("Cells in rest area")

    ax[1][0].imshow(grans)
    ax[1][0].imshow(Anc, cmap=plt.cm.gray, alpha=0.5)
    ax[1][1].imshow(grans)
    ax[1][1].imshow(Arc, cmap=plt.cm.gray, alpha=0.5)
    plt.show()

    #Now we need to calculate the areas of those cells in regard to
    #the total areas of those regions, so we will find the contours in
    #those and get their area
    #An area
    contours_An = ski.measure.find_contours(An, mask=mask_keep)
    contours_Ar = ski.measure.find_contours(Ar, mask=mask_keep)
    contours_Anc = ski.measure.find_contours(Anc)   #mask=mask_keep here does not generate closed contours, so we don't want it
    contours_Arc = ski.measure.find_contours(Arc)
    print("[NOTE] Areas are calculated through pixel counting which is more accurate than contour calculation, however for the visualizations and for the area distribution the contour method is used.")
    print("[+] Visualizing contours for area calculation")
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    fig.canvas.manager.set_window_title("Contours visualization for area calculation")
    ax[0][0].imshow(An, cmap=plt.cm.gray)
    ax[0][0].set_title("FTS-Neighbourhood")
    for contour in contours_An:
        ax[0][0].plot(contour[:,1], contour[:,0], color='red')
    ax[0][1].imshow(Ar, cmap=plt.cm.gray)
    ax[0][1].set_title("Rest area")
    for contour in contours_Ar:
        ax[0][1].plot(contour[:,1], contour[:,0], color='red')
    ax[1][0].imshow(Anc, cmap=plt.cm.gray)
    for contour in contours_Anc:
        ax[1][0].plot(contour[:,1], contour[:,0], color='red')
    ax[1][1].imshow(Arc, cmap=plt.cm.gray)
    for contour in contours_Arc:
        ax[1][1].plot(contour[:,1], contour[:,0], color='red')
    plt.show()

    #simply count the number of white pixels
    print("[+] Calculating regions area")
    Area_image = grans.shape[0]*grans.shape[1]
    Area_mask_keep = np.count_nonzero(~mask_keep.astype('bool'))
    Area_At = Area_image - Area_mask_keep

    Area_An = np.count_nonzero(An.astype('bool'))

    Area_Ar = Area_At - Area_An
    print("[OUTPUT] Areas of whole image, At, An and Ar are:\n{d}\n{a}\n{b}\n{c}".format(d=Area_image,a=Area_At, b=Area_An, c=Area_Ar))

    #now for the areas of the cells
    areas_neighbourhood_cells = []
    areas_rest_cells = []
    for j in contours_Anc:
        areas_neighbourhood_cells.append(PolyArea(j[:,1], j[:,0]))
    for j in contours_Arc:
        areas_rest_cells.append(PolyArea(j[:,1], j[:,0]))
    
    Area_Anc = np.count_nonzero(Anc.astype('bool'))
    Area_Arc = np.count_nonzero(Arc.astype('bool'))
    print("[OUTPUT] Areas of cells in neighbourhoods (Anc) and of cells in rest area (Arc) are:\n{a}\n{b}".format(a=Area_Anc, b=Area_Arc))

    print("[+] Visualizing area distribution (based on contour calculation) in boxplots.")
    total_areas = areas_neighbourhood_cells.copy()
    total_areas.extend(areas_rest_cells)
    fig, ax = plt.subplots(1,3)
    fig.canvas.manager.set_window_title("Cell areas distribution")
    ax[0].boxplot(areas_neighbourhood_cells)
    ax[0].set_title("Anc")
    ax[1].boxplot(areas_rest_cells)
    ax[1].set_title("Arc")
    ax[2].boxplot(total_areas)
    ax[2].set_title("All cells")
    fig.tight_layout()
    plt.show()

    #now for fractions
    fN = Area_Anc / Area_An
    fR = Area_Arc / Area_Ar
    print("[OUTPUT] Percentage of area of cells over area of region is:\n\tAnc\t{a}\n\tArc\t{b}".format(a=fN*100,b=fR*100))
    print("[i] There are {a} cells in the fts-neighbourhoods and {b} cells in the rest area".format(a=len(contours_Anc), b=len(contours_Arc)))
    
    #saving outside of loop
    fN_list.append(fN)
    fR_list.append(fR)
    total_areas = np.array(total_areas)
    average_area = total_areas.mean()
    average_area_of_all_cells_list.append(average_area)
    
    #run again?
    print("[+] Sample run successfully completed.")
    counter += 1
    confirmation = input("[INPUT] If you want to run again with another sample type y, else type n, then press Enter: ")
    if confirmation.strip().lower() == 'n':
        break
    elif confirmation.strip().lower() == 'y':
        sample_flag = True

print("[+] You have analysed {n} samples.".format(n=len(fN_list)))
if len(fN_list) == 1:
    print("[i] Stasticial analysis of samples cannot be completed with 1 sample. However, we still can simulate random cell positions and see where does your sample fit in this distribution, \
if you do not want to do this just exit the program.")

print("[+] N images with white pixels (at random locations each time) of size equal to that of the first image (height={h}, width={w}) will be created.".format(h=first_image_height, w=first_image_width))
number_of_images = input("[INPUT] type N and press Enter: ")
number_of_images = int(float(number_of_images))
seeds_number = []
flag = True
while flag:
    flag = False
    seeds = input("[INPUT] number of random cells and press Enter, if done type q and press Enter: ")
    if seeds.strip().lower() == 'q':
        break
    else:
        seeds = int(float(seeds))
        seeds_number.append(seeds)
        flag = True
print("[+] There will be created {N} images for each of the following number of white pixels:\n{S}".format(N=number_of_images, S=seeds_number))

#we will need to aply At mask and Abn mask and also neighbourhood area
#plus we will need to recalculate their areas
print("[+] On each of the generated images, there will be applied a background-ignore mask (exactly like the first one from before) \
and a mask to generate the neighbourhood from (exactly like the second mask from before), also a neighbourhood dilation kernel size \
will be needed.")
mask_path = input("[INPUT] first mask's path and press Enter: ")
mask = ski.io.imread(mask_path)
mask = mask[:,:,:3]
mask_g = mask[:,:,1]
mask_keep = ~(mask_g/255).astype('int').astype('bool')  #keep = True
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Mask - At")
ax.imshow(mask_keep, cmap=plt.cm.gray); plt.show()

ft_mask_path = input("[INPUT] second mask's path and press Enter: ")
ft_mask = ski.io.imread(ft_mask_path)
ft_mask = ft_mask[:,:,:3]
ft_mask_g = ft_mask[:,:,1]
ft_mask_keep = (ft_mask_g/255).astype('int').astype('bool') #fts are True are keep
fig, ax = plt.subplots(); fig.canvas.manager.set_window_title("Mask - Af")
ax.imshow(ft_mask_keep, cmap=plt.cm.gray); plt.show()

print("[+] The same way as before you will need to setup the neighbourhood dilation kernel (note that you may enter the parameters only once, not many times like before).")
k_width = input("[INPUT] dilation kernel's width and press Enter: ")
k_height = input("[INPUT] dilation kernel's height and press Enter: ")
k_width = int(float(k_width)); k_height = int(float(k_height))
dilation_kernel = create_dilation_kernel(k_height, k_width)

#Abn and An and Ar will be calculated here since they don't involve the image in any way
Abn = ski.morphology.dilation(ft_mask_keep, dilation_kernel).astype('int')
An = Abn - ft_mask_keep.astype(Abn.dtype)
An = An*mask_keep

Ar = ~Abn.astype('bool')
Ar = Ar * mask_keep #although unnecessary it is only once, so it won't hurt
#Ar is pretty much the same as Ar_for_view, however it is useless
#since everything regarding it is calculated by Abn

#now we need to calculate region areas (the same way as above)
Area_image = first_image_height*first_image_width
Area_mask_keep = np.count_nonzero(~mask_keep.astype('bool'))
Area_At = Area_image - Area_mask_keep

Area_An = np.count_nonzero(An.astype('bool'))

Area_Ar = Area_At - Area_An
print("[OUTPUT] Areas of whole image, At, An and Ar are:\n{d}\n{a}\n{b}\n{c}".format(d=Area_image,a=Area_At, b=Area_An, c=Area_Ar))


#it is really bad practice to generate and store all images and then analyse them
#on each image we need to randomly enlarge the pixels until their area distibution
#is like the total cell area distribution, then for each image count the number of
#cells in the neighbourhood region and the number of cells outside of it and then
#calculate their percentage

expansion_kernels = [np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.uint8),
                        np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=np.uint8),
                        np.array([[0,0,1],[0,1,0],[0,0,0]], dtype=np.uint8),
                        np.array([[0,0,0],[1,1,0],[0,0,0]], dtype=np.uint8),
                        np.array([[0,0,0],[0,1,1],[0,0,0]], dtype=np.uint8),
                        np.array([[0,0,0],[0,1,0],[1,0,0]], dtype=np.uint8),
                        np.array([[0,0,0],[0,1,0],[0,1,0]], dtype=np.uint8),
                        np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.uint8),]
                        #np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.uint8),   does not expand, don't use

av_area_threshold = sum(average_area_of_all_cells_list)/len(average_area_of_all_cells_list)
print("[+] Average area to be reached is {a} in pixels^2.".format(a=av_area_threshold))
fN_randoms_list = []
fR_randoms_list = []
print("[+] Processing random images, this will take some time, go take a break.")
for seeds in tqdm(seeds_number, position=0):
    for i in tqdm(range(number_of_images), position=1, leave=False):
        image = np.zeros(first_image_height*first_image_width, dtype='bool')
        image[:seeds] = True
        np.random.shuffle(image)
        image = image.reshape((first_image_height, first_image_width))
        #random expansion until threshold
        av_area = 0
        while av_area < av_area_threshold:
            kernel = random.choice(expansion_kernels)
            contours = ski.measure.find_contours(image)
            areas = [PolyArea(j[:,1], j[:,0]) for j in contours]
            av_area = sum(areas)/len(areas)
            if av_area >= av_area_threshold:
                break
            else:
                image = convolve2d(image, kernel, mode='same')
                image = image.astype('bool')
        #now we need to apply the masks in the following order
        #first the At mask
        At_image = image * mask_keep
        At_image = At_image.astype('bool')
        #now apply the dilation kernels to find Anc and Arc
        #here At_image is the equivalent of masked_image form before
        Anc = An * At_image
        Anc = Anc.astype('bool')    #although everything is white from At_image being bool, another bool won't hurt to be sure
        Arc = Ar * At_image
        Arc = Arc.astype('bool')
        #now for the calculations
        Area_Anc = np.count_nonzero(Anc.astype('bool'))
        Area_Arc = np.count_nonzero(Arc.astype('bool'))
        
        fN = Area_Anc / Area_An
        fR = Area_Arc / Area_Ar
        
        fN_randoms_list.append(fN)
        fR_randoms_list.append(fR)

print("[+] Random simulations completed.")
print("[+] Visualizing fN and fR as boxplots.")
fig, ax = plt.subplots(1,2)
fig.canvas.manager.set_window_title("Random cell fraction areas distributions")
ax[0].boxplot(fN_randoms_list)
ax[0].set_title("fN")
ax[1].boxplot(fR_randoms_list)
ax[1].set_title("fR")
fig.tight_layout()
plt.show()

if (len(fN_list) == 1) or ((len(set(fN_list)) == 1) and (len(set(fR_list)) == 1)):
    if len(fN_list) == 1:
        print("[+] You have only analysed 1 sample.")
    else:
        print("[+] Although you have analysed {n} samples, they all return the same fN and fR values, so a regression plot of them cannot be done, hence we will treat them as a single point.".format(n=len(fN_list)))
    print("[+] Creating normalized (0-1) quantile-quantile plot of random data and calculating appropriate statistics.")
    
    #normalizing data in range 0-1
    fN_randoms_list = np.array(fN_randoms_list)
    fR_randoms_list = np.array(fR_randoms_list)
    fN_randoms_normalized = (fN_randoms_list-fN_randoms_list.min())/(fN_randoms_list.max()-fN_randoms_list.min())
    fR_randoms_normalized = (fR_randoms_list-fR_randoms_list.min())/(fR_randoms_list.max()-fR_randoms_list.min())
    #now normalize point with x based on fN and y based on fR
    sample_fN = fN_list[0]
    sample_fR = fR_list[0]
    sample_fN_normed = (sample_fN-fN_randoms_list.min())/(fN_randoms_list.max()-fN_randoms_list.min())
    sample_fR_normed = (sample_fR-fR_randoms_list.min())/(fR_randoms_list.max()-fR_randoms_list.min())
    
    #now enter quantiles loop
    flag = True
    while flag:
        flag = False
        max_quantile = input("[INPUT] how many quantiles do you want to calculate (e.g. 10 for 0.1,0.2,...,0.9,1.0): ")
        max_quantile = int(float(max_quantile))
        
        quantiles_fN_randoms = [np.quantile(fN_randoms_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        quantiles_fR_randoms = [np.quantile(fR_randoms_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        #regression line with fN quantiles on x_axis and fR quantiles on y_axis
        #1st degree polynomial regression
        slope, intercept = np.polyfit(quantiles_fN_randoms, quantiles_fR_randoms, 1)
        #compute distances
        sample_dist = distance_of_point_from_line(slope, intercept, sample_fN_normed, sample_fR_normed)
        random_dists = [distance_of_point_from_line(slope, intercept, quantiles_fN_randoms[i], quantiles_fR_randoms[i]) for i in range(len(quantiles_fN_randoms))]
        #output stuff and then plot
        print("[OUTPUT] original sample data was\n\tfN={a}, fR={b}\nnormalized sample data is\n\tfN={c}, fR={d}".format(a=sample_fN,b=sample_fR,c=sample_fN_normed,d=sample_fR_normed))
        print("[OUTPUT] 1st deg polynomial regression line of random quantiles equation is:\n\ty={a}*x+({b})".format(a=slope,b=intercept))
        print("[OUTPUT] distance of sample point from above line is\n\t{d}".format(d=sample_dist))
        #seaborn regplot left, boxplot right
        fig, ax = plt.subplots(1,2)
        fig.canvas.manager.set_window_title("Statistical relevance plots of sample and random simulations")
        sns.regplot(x=quantiles_fN_randoms,y=quantiles_fR_randoms,color='blue',marker='x', ax=ax[0], label='Simulations')
        ax[0].scatter(sample_fN_normed, sample_fR_normed, color='red')
        ax[0].set_title("Q-Q plot")
        ax[0].set(xlabel='fN', ylabel='fR')
        ax[0].legend()
        
        ax[1].boxplot(random_dists)
        ax[1].scatter(1,sample_dist, color='red')
        ax[1].set_title("Residual distances")
        plt.show()
        
        #ask if try again
        confirmation = input("[INPUT] if you want to try again with different quantiles type 'y' and press Enter: ")
        if confirmation.strip().lower() == 'y':
            flag = True
elif len(fN_list) >= 2: #now we can do two quantile-quantile plots, although at least n samples for n quantiles is prefered
    print("[+] You have analysed {n} samples.".format(n=len(fN_list)))
    print("[+] Creating quantile-quantile plots and calculating appropriate statistics.")
    
    #normalizing data in range 0-1
    fN_randoms_list = np.array(fN_randoms_list)
    fR_randoms_list = np.array(fR_randoms_list)
    fN_list = np.array(fN_list)
    fR_list = np.array(fR_list)
    
    fN_randoms_normalized = (fN_randoms_list-fN_randoms_list.min())/(fN_randoms_list.max()-fN_randoms_list.min())
    fR_randoms_normalized = (fR_randoms_list-fR_randoms_list.min())/(fR_randoms_list.max()-fR_randoms_list.min())
    fN_list_normalized = (fN_list - fN_list.min())/(fN_list.max()-fN_list.min())
    fR_list_normalized = (fR_list - fR_list.min())/(fR_list.max()-fR_list.min())
    
    #now enter quantiles loop
    flag = True
    while flag:
        flag = False
        max_quantile = input("[INPUT] how many quantiles do you want to calculate (e.g. 10 for 0.1,0.2,...,0.9,1.0): ")
        max_quantile = int(float(max_quantile))
        
        quantiles_fN_randoms = [np.quantile(fN_randoms_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        quantiles_fR_randoms = [np.quantile(fR_randoms_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        quantiles_fN = [np.quantile(fN_list_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        quantiles_fR = [np.quantile(fR_list_normalized, i/max_quantile) for i in range(1,max_quantile+1)]
        
        
        if (len(set(fN_list)) > 1):
            #regression line with fN on x-axis and fR on y-axis
            #1st degree polynomial regression
            slope_random, intercept_random = np.polyfit(quantiles_fN_randoms, quantiles_fR_randoms, 1)
            slope, intercept = np.polyfit(quantiles_fN, quantiles_fR, 1)
            #compute distances
            random_dists = [distance_of_point_from_line(slope_random, intercept_random,quantiles_fN_randoms[i],quantiles_fR_randoms[i]) for i in range(len(quantiles_fN_randoms))]
            samples_dists = [distance_of_point_from_line(slope, intercept, quantiles_fN[i], quantiles_fR[i]) for i in range(len(quantiles_fN))]
            #output stuff and then plot
            print("[OUTPUT] 1st deg polynomial regression line of SAMPLES' quantiles equation is:\n\ty={a}*x+({b})".format(a=slope,b=intercept))
            print("[OUTPUT] 1st deg polynomial regression line of RANDOM quantiles equation is:\n\ty={a}*x+({b})".format(a=slope_random,b=intercept_random))
            #seaborn regplots left, samples boxplot middle, random boxplot right
            fig, ax = plt.subplots(1,3)
            fig.canvas.manager.set_window_title("Statistical relevance plots of samples and random simulations")
            sns.regplot(x=quantiles_fN_randoms,y=quantiles_fR_randoms,color='blue',marker='x',ax=ax[0],label="Simulations")
            sns.regplot(x=quantiles_fN,y=quantiles_fR,color='red',marker='x',ax=ax[0],label="Samples")
            ax[0].set_ybound(0,2)
            ax[0].set_title("Q-Q plot")
            ax[0].set(xlabel="fN", ylabel="fR")
            ax[0].legend()
            
            ax[1].boxplot(samples_dists)
            ax[1].set_title("Samples res. distances")
            
            ax[2].boxplot(random_dists)
            ax[2].set_title("Random res. distances")
            plt.show()
        elif (len(set(fN_list)) == 1):
            print("[+] Note that all fN values calculated from the samples are the same, while the fR values aren't, so in order to attempt to do a regression plot, we will plot fN on y-axis and fR on x-axis.")
            #regression line with fN on y-axis and fR on x-axis
            #1st degree polynomial regression
            slope_random, intercept_random = np.polyfit(quantiles_fR_randoms, quantiles_fN_randoms, 1)
            slope, intercept = np.polyfit(quantiles_fR, quantiles_fN, 1)
            #compute distances
            random_dists = [distance_of_point_from_line(slope_random, intercept_random,quantiles_fR_randoms[i],quantiles_fN_randoms[i]) for i in range(len(quantiles_fR_randoms))]
            samples_dists = [distance_of_point_from_line(slope, intercept, quantiles_fR[i], quantiles_fN[i]) for i in range(len(quantiles_fR))]
            #output stuff and then plot
            print("[OUTPUT] 1st deg polynomial regression line of SAMPLES' quantiles equation is:\n\ty={a}*x+({b})".format(a=slope,b=intercept))
            print("[OUTPUT] 1st deg polynomial regression line of RANDOM quantiles equation is:\n\ty={a}*x+({b})".format(a=slope_random,b=intercept_random))
            #seaborn regplots left, samples boxplot middle, random boxplot right
            fig, ax = plt.subplots(1,3)
            fig.canvas.manager.set_window_title("Statistical relevance plots of samples and random simulations")
            sns.regplot(x=quantiles_fR_randoms,y=quantiles_fN_randoms,color='blue',marker='x',ax=ax[0],label="Simulations")
            sns.regplot(x=quantiles_fR,y=quantiles_fN,color='red',marker='x',ax=ax[0],label="Samples")
            ax[0].set_ybound(0,2)
            ax[0].set_title("Q-Q plot")
            ax[0].set(xlabel="fR", ylabel="fN")
            ax[0].legend()
            
            ax[1].boxplot(samples_dists)
            ax[1].set_title("Samples res. distances")
            
            ax[2].boxplot(random_dists)
            ax[2].set_title("Random res. distances")
            plt.show()
            
        #ask if try again
        confirmation = input("[INPUT] if you want to try again with different quantiles type 'y' and press Enter: ")
        if confirmation.strip().lower() == 'y':
            flag = True