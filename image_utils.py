import cv2
import numpy as np
import matplotlib.pyplot as plt

image_size=300
def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x,size):
    """Generates bounding box array from a train_df row"""
    return np.array([int(x[3]),int(x[2]),int(x[5]),int(x[4])])

def resize_image_bb(read_path,bb,size):
    """Resize an image and its bounding box and write image to new path"""
    im = cv2.cvtColor(cv2.imread(str(read_path)), cv2.COLOR_BGR2RGB)
    im_resized = cv2.resize(im, (size, size))
    Y_resized = cv2.resize(create_mask(bb, im), (size, size))
    #new_path = str(write_path/Path(read_path).parts[-1])
    new_path=str(read_path[:-4]+'resized'+ read_path[-4:])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y


def transformsXY(path, bb):
    x = cv2.imread(str(path))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
    Y = create_mask(bb, x)

    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]