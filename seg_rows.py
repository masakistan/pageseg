import cv2, sys
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from os import makedirs

MEAN_THRESH = 250
MIN_TEXT_HEIGHT = 10
MERGE_THRESH = 1
FILTER_THRESH = 3
HORIZONTAL_SLICE = 0
EXPECTED_HEIGHT = 16
MERGED_ROW_TOLERANCE = 5

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

    
def filter(z, p, v, z_thresh):
    n_z, n_p, n_v = [], [], []
    for _z, _p, _v in zip(z, p, v):
        if _z > z_thresh:
            n_z.append(_z)
            n_p.append(_p)
            n_v.append(_v)
    return np.array(n_z), np.array(n_p), np.array(n_v)


def merge(z, p, thresh):
    n_z, n_p = [], []

    s_z, s_p = z[0], p[0]
    for i in range(1, len(z)):
        #print('current:', s_p, s_z)
        if abs(s_p + s_z - p[i]) < thresh:
            s_z += z[i]
        else:
            n_z.append(s_z)
            n_p.append(s_p)
            s_z = z[i]
            s_p = p[i]
            
    n_z.append(s_z)
    n_p.append(s_p)
 
    return np.array(n_z), np.array(n_p)

def split_erroneously_merged_rows(z, p):
    n_z, n_p = [], []
    for _z, _p in zip(z, p):
        #print(_p, _z)
        if _z > EXPECTED_HEIGHT + MERGED_ROW_TOLERANCE:
            #if _z % EXPECTED_HEIGHT < MERGED_ROW_TOLERANCE:
            factor = _z // EXPECTED_HEIGHT
            if _z % EXPECTED_HEIGHT > EXPECTED_HEIGHT - MERGED_ROW_TOLERANCE:
                #factor = (_z // EXPECTED_HEIGHT) + 1
                factor += 1
            print("found row to be split", _p, _z, 'factor:', factor)
            #print(factor)
            for i in range(factor):
                height = _z // factor
                start = _p + (i * height)
                print(_p, start, _z, height)
                n_z.append(height)
                n_p.append(start)
        else:
            n_z.append(_z)
            n_p.append(_p)
    return n_z, n_p

    
out_dir = sys.argv[1]

try:
    makedirs(out_dir)
except:
    print('dir already exists')

# NOTE: read in the image
img = cv2.imread(sys.argv[2])

# NOTE: convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# NOTE: binarize image
img_gray[img_gray >= 200] = 255
img_gray[img_gray < 200] = 0
binarized_out_path = join(out_dir, 'binarized.jpg')
print(binarized_out_path)
cv2.imwrite(binarized_out_path, img_gray)

# NOTE: keep a copy of the original file
img_gray_orig = img_gray

img_gray = img_gray[:, HORIZONTAL_SLICE :]

# NOTE: try to figure out binarization thresholds by looking at histogram of grayscale pixel values
#plt.hist(img_gray.flatten(), bins = 'auto')
#plt.show()

# NOTE: plot distribution of active pixels across the x axis
#dist = np.array(img_gray_orig, dtype = bool)
#dist[dist == 255] = 1
#dist = np.invert(dist)
#dist = np.sum(dist, axis = 0)
#plt.bar([i for i in range(len(dist))], dist)
#plt.show()

means = np.min(img_gray, axis = 1)

# NOTE: line
#plt.plot(means)
#plt.show()

#for i, val in enumerate(sums):
#    print(i, val)


boundaries = means > MEAN_THRESH

#for i, val in enumerate(boundaries):
#    print(i, val)


z, p, v = rle(boundaries)
# NOTE: clean boundaries
z, p, v = filter(z, p, v, FILTER_THRESH)
v = v != True

#for i in zip(z, p, v):
#    print(i)
    
#print('*' * 20)

z = z[v]
p = p[v]

z, p = merge(z, p, MERGE_THRESH)
z, p = split_erroneously_merged_rows(z, p)

#for i in zip(z, p):
#    print(i)

config = '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789'
total_snippets = 0
for idx, (start, length) in enumerate(zip(p, z)):
    if length < MIN_TEXT_HEIGHT:
        continue
    
    out_path = join(out_dir, str(idx) + '.jpg')
    snippet = img_gray_orig[start - 2: start + length + 2, :]
    cv2.imwrite(out_path, snippet)
    snippet = cv2.resize(snippet, tuple([x * 2 for x in snippet.shape[::-1]]))
    #print(idx, ':', pytesseract.image_to_string(snippet, config = config))
    
    #print('out path:', out_path)
    total_snippets += 1
    
print("total snippets:", total_snippets)

# NOTE: Histogram
#plt.hist(sums, bins = 'auto')
#plt.show()

