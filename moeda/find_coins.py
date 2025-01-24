import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
#from skimage.morphology import closing, disk, erosion
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.segmentation import watershed, slic
from scipy import ndimage as ndi

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(9, 9),
                        layout="constrained")

moeda_100 = []
for i in range(21,94):
    moeda_100.append("br_coins_train_test/train/100/100-back ("+str(i)+").JPG")
    
for i in range(19,81):
    moeda_100.append("br_coins_train_test/train/100/100-front ("+str(i)+").JPG")

for i in range(len(moeda_100)):
    img = io.imread(moeda_100[i])

    img_cinza = rgb2gray(img)
            
    edges = canny(img_cinza,sigma=5)

    # Detect two radii
    #hough_radii = np.arange(190, 250, 2) #-> Raio para Moeda de 50 centavos
    hough_radii = np.arange(260, 265, 2) #-> Raio para Moeda de 100 centavos
    hough_res = hough_circle(edges, hough_radii)
  
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=25)
    img_mod = img.copy()
    cx = int(sum(cx)/len(cx))
    cy = int(sum(cy)/len(cy))
    radii = int(sum(radii)/len(radii))
    
    img_novotamanho = cv.resize(img[cy-radii:cy+radii,cx-radii:cx+radii,:], (415,415))
    
    io.imsave("br_coins_train_test/train/100_corte/corte_"+str(i)+".JPG", img_novotamanho)
