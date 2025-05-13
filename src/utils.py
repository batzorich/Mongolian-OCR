import cv2
import matplotlib.pyplot as plt 

def display_image_with_axes(im_data, dpi=100):

    if len(im_data.shape) == 3:
        height, width, depth = im_data.shape
    else:
        height, width = im_data.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im_data, cmap='gray')
    plt.show()

def reshape_image(image, scale_index):
    height, width = image.shape[:2]
    
    new_width = int(width / scale_index)
    
    resized_image = cv2.resize(image, (new_width, height))
    
    return resized_image

def count_consecutive(bool_list):
    result = []
    i = 0
    n = len(bool_list)
    while i < n:
        val = bool_list[i]
        count = 0
        while i < n and bool_list[i] == val:
            count += 1
            i += 1
        result.append((count, 0) if not val else (0, count))
    return result