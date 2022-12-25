import cv2 #görüntü işleme
import sys #Python sürümü ile işlemler
import kdimensionaltree
import init
import numpy as np
import config as cnfg
from time import time
from scipy import ndimage
from sklearn.decomposition import PCA

def BoundingBox(mask):
    start = time()
    a = np.where(mask != 0)
    box = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cnfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (box[2], box[0]), (box[3], box[1]), (255,255,255), 1) 
        cv2.imwrite(cnfg.OUT_FOLDER + cnfg.IMAGE + cnfg.BB_IMAGE_SUFFIX, mask)
    end = time()
    print ("BoundingBox execution time: ", end - start)
    return box


def SearchDomain(shape, box):
    start = time()
    column_min, column_max = max(0, 2*box[0] - box[1]), min(2*box[1] - box[0], shape[1]-1)    
    row_min, row_max = max(0, 2*box[2] - box[3]), min(2*box[3] - box[2], shape[0]-1) 
    end = time()
    print ("SearchDomain execution time: ", end - start)
    return column_min, column_max, row_min, row_max



def Patches(image, box, hole):
    start = time()
    indices, patches = [], []
    rows, columns, _ = image.shape
    for i in range(box[2]+cnfg.PATCH_SIZE//2, box[3]-cnfg.PATCH_SIZE//2):
        for j in range(box[0]+cnfg.PATCH_SIZE//2, box[1]-cnfg.PATCH_SIZE//2):
            if i not in range(hole[2]-cnfg.PATCH_SIZE//2, hole[3]+cnfg.PATCH_SIZE//2) and j not in range(hole[0]-cnfg.PATCH_SIZE//2, hole[1]+cnfg.PATCH_SIZE//2):
                indices.append([i,j])
                patches.append(image[i-cnfg.PATCH_SIZE//2:i+cnfg.PATCH_SIZE//2, j-cnfg.PATCH_SIZE//2:j+cnfg.PATCH_SIZE//2].flatten())
    end = time()
    print ("Patches execution time: ", end - start)
    return np.array(indices), np.array(patches, dtype='int64')



def ReduceDimension(patches):
    start = time()
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print ("ReduceDimension execution time: ", end - start)
    return reducedPatches



def Offsets(patches, indices):
    start = time()
    kd = kdimensionaltree.KDTree(patches, leafsize=cnfg.KDT_LEAF_SIZE, tau=cnfg.TAU)
    dist, offsets = kdimensionaltree.get_annf_offsets(patches, indices, kd.tree, cnfg.TAU)
    end = time()
    print ("Offsets execution time: ", end - start)
    return offsets


def KDominantOffsets(offsets, K, height, width):
    start = time()
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    bins = [[i for i in range(np.min(x),np.max(x))], [i for i in range(np.min(y),np.max(y))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    p, q = np.where(hist == cv2.dilate(hist, np.ones(8)))
    nonMaxSuppressedHist = np.zeros(hist.shape)
    nonMaxSuppressedHist[p, q] = hist[p, q]
    p, q = np.where(nonMaxSuppressedHist >= np.partition(nonMaxSuppressedHist.flatten(), -K)[-K])
    peakHist = np.zeros(hist.shape)
    peakHist[p, q] = nonMaxSuppressedHist[p, q]
    peakOffsets, freq = [[xedges[j], yedges[i]] for (i, j) in zip(p, q)], nonMaxSuppressedHist[p, q].flatten()
    peakOffsets = np.array([x for _, x in sorted(zip(freq, peakOffsets), reverse=True)], dtype="int64")[:2*K]
    end = time()
    print ("KDominantOffsets execution time: ", end - start)
    return peakOffsets 


def OptimizedLabels(image, mask, labels):
    start = time()
    optimizer = init.Optimizer(image, mask, labels)
    sites, optimalLabels = optimizer.InitializeLabelling()
    optimalLabels = optimizer.OptimizeLabellingABS(optimalLabels)
    end = time()
    print ("OptimizedLabels execution time: ", end - start)
    return sites, optimalLabels 
# bir görüntüyü, bir maskeyi ve bir dizi etiketi girdi olarak alan ve iki öğe içeren bir demet döndüren bir işlev


def CompleteImage(image, sites, mask, offsets, optimalLabels):
    failedPoints = mask
    completedPoints = np.zeros(image.shape)
    finalImg = image
    for i in range(len(sites)):
        j = optimalLabels[i]
        finalImg[sites[i][0], sites[i][1]] = image[sites[i][0] + offsets[j][0], sites[i][1] + offsets[j][1]]
        completedPoints[sites[i][0], sites[i][1]] = finalImg[sites[i][0], sites[i][1]]
        failedPoints[sites[i][0], sites[i][1]] = 0
    return finalImg, failedPoints, completedPoints


def PoissonBlending(image, mask, center):
    src = cv2.imread(cnfg.OUT_FOLDER + cnfg.IMAGE + "_CompletedPoints.png")
    dst = cv2.imread(cnfg.OUT_FOLDER + cnfg.IMAGE + ".png")
    blendedImage = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    return blendedImage


def main(imageFile, maskFile):
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    imageR = cv2.imread(imageFile)
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = BoundingBox(mask)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cnfg.TAU = max(bbwidth, bbheight)/15
    cnfg.DEFLAT_FACTOR = image.shape[1]
    sd = SearchDomain(image.shape, bb)
    indices, patches = Patches(imageR, sd, bb)
    reducedPatches = ReduceDimension(patches)
    offsets = Offsets(reducedPatches, indices)
    kDominantOffset = KDominantOffsets(offsets, 60, image.shape[0], image.shape[1])
    sites, optimalLabels = OptimizedLabels(imageR, mask, kDominantOffset)
    completedImage, failedPoints, completedPoints = CompleteImage(imageR, sites, mask, kDominantOffset, optimalLabels)
    cv2.imwrite(cnfg.OUT_FOLDER + cnfg.IMAGE + "_Complete.png", completedImage)
    cv2.imwrite(cnfg.OUT_FOLDER + cnfg.IMAGE + "_CompletedPoints.png", completedPoints)
 #   center = (bb[2]+bbwidth//2, bb[0]+bbheight//2)
 #   blendedImage = PoissonBlending(imageR, mask,center)
 #   cv2.imwrite(cnfg.OUT_FOLDER + cnfg.IMAGE + "_blendedImage.png", blendedImage)

    if (np.sum(failedPoints)):
        cv2.imwrite(cnfg.OUT_FOLDER + cnfg.IMAGE + "_Failed.png", failedPoints)
        main(cnfg.OUT_FOLDER + cnfg.IMAGE + "_Complete.png", cnfg.OUT_FOLDER + cnfg.IMAGE + "_Failed.png")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanim: python main.py image_name mask_file_name")
        exit()
    cnfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cnfg.SRC_FOLDER + sys.argv[1]
    print(imageFile)
    maskFile = cnfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    