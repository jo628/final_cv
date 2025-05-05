
# -*- coding: utf-8 -*-
"""
Selective Search implementation based on paper
"Selective Search for Object Recognition" by J.R.R. Uijlings et al.
Simplified for waste detection usage.
"""

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def _calculate_texture_gradient(img):
    """
    Calculate texture gradient for entire image
    The original implementation used 8 Gabor filter kernels
    This version uses simpler gradient calculations for speed
    """
    # Convert to grayscale
    gx = np.absolute(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.absolute(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    
    # Calculate magnitude and direction
    mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    
    # Normalize to 0-255
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mag

def _sim_colour(r1, r2):
    """
    Calculate color similarity
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

def _sim_texture(r1, r2):
    """
    Calculate texture similarity
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

def _sim_size(r1, r2, imsize):
    """
    Calculate size similarity
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize

def _sim_fill(r1, r2, imsize):
    """
    Calculate fill similarity
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def _calc_sim(r1, r2, imsize):
    """
    Calculate full similarity
    """
    return _sim_colour(r1, r2) + _sim_texture(r1, r2) + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize)

def _calc_colour_hist(img):
    """
    Calculate color histogram for a region
    """
    BINS = 25
    hist = np.array([])
    
    for colour_channel in (0, 1, 2):
        c = img[:, colour_channel]
        hist = np.concatenate(
            [hist, np.histogram(c, BINS, (0.0, 255.0))[0]]
        )
    
    # Normalize
    hist = hist / len(img)
    
    return hist

def _calc_texture_gradient(img):
    """
    Calculate texture gradient for img
    returns gradient histograms for 4 directions
    """
    ret = np.zeros((4, img.shape[0], img.shape[1]))
    
    # Calculate gradients
    ret[0] = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    ret[1] = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    ret[2] = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3))
    ret[3] = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, -1, ksize=3))
    
    return ret

def _calc_texture_hist(img):
    """
    Calculate texture histogram for a region
    """
    BINS = 10
    
    hist = np.array([])
    
    grad = _calc_texture_gradient(img)
    
    for i in range(4):
        for j in range(3):
            channel = grad[i][:, j]
            hist = np.concatenate(
                [hist, np.histogram(channel, BINS, (0.0, 255.0))[0]]
            )
    
    # Normalize
    hist = hist / len(img)
    
    return hist

def _extract_regions(img):
    """
    Extract initial regions using simple segmentation
    """
    import cv2
    
    # Convert to float
    img_float = img.astype(np.float64)
    
    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_float, (3, 3), 0)
    
    # Apply simple Felzenszwalb segmentation with default parameters
    segments = skimage.segmentation.felzenszwalb(
        img_blur, scale=100, sigma=0.5, min_size=50
    )
    
    # Calculate region properties
    regions = []
    
    # Get region masks and calculate properties
    for i in np.unique(segments):
        mask = (segments == i)
        region = {}
        region["mask"] = mask
        region["size"] = np.sum(mask)
        region["labels"] = [i]
        
        # Bounding box
        y, x = np.where(mask)
        region["min_x"] = np.min(x)
        region["max_x"] = np.max(x)
        region["min_y"] = np.min(y)
        region["max_y"] = np.max(y)
        
        # Extract color and texture histograms
        masked_pixels = img[mask]
        if len(masked_pixels) > 0:
            region["hist_c"] = _calc_colour_hist(masked_pixels)
            region["hist_t"] = _calc_texture_hist(masked_pixels)
            regions.append(region)
    
    return regions, segments

def _merge_regions(regions, similarities, imsize):
    """
    Merge similar regions iteratively
    """
    # Sort similarities in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Merge similar regions
    new_regions = []
    
    # Used to check if a region was already merged
    merged = [False] * len(regions)
    
    # Merge similar regions
    for sim, i, j in similarities:
        # Check if regions were already merged
        if merged[i] or merged[j]:
            continue
        
        # Create new region
        new_region = {
            "labels": regions[i]["labels"] + regions[j]["labels"],
            "size": regions[i]["size"] + regions[j]["size"],
            "mask": regions[i]["mask"] | regions[j]["mask"],
            "min_x": min(regions[i]["min_x"], regions[j]["min_x"]),
            "max_x": max(regions[i]["max_x"], regions[j]["max_x"]),
            "min_y": min(regions[i]["min_y"], regions[j]["min_y"]),
            "max_y": max(regions[i]["max_y"], regions[j]["max_y"]),
            "hist_c": (regions[i]["hist_c"] * regions[i]["size"] + regions[j]["hist_c"] * regions[j]["size"]) / (regions[i]["size"] + regions[j]["size"]),
            "hist_t": (regions[i]["hist_t"] * regions[i]["size"] + regions[j]["hist_t"] * regions[j]["size"]) / (regions[i]["size"] + regions[j]["size"])
        }
        
        # Add new region
        new_regions.append(new_region)
        
        # Mark regions as merged
        merged[i] = True
        merged[j] = True
    
    # Add non-merged regions
    for i, m in enumerate(merged):
        if not m:
            new_regions.append(regions[i])
    
    return new_regions

def selective_search(img, scale=100, sigma=0.8, min_size=50):
    """
    Selective search for object proposals
    """
    import cv2
    
    # Extract initial regions
    regions, img_segments = _extract_regions(img)
    
    # Calculate similarities
    similarities = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            sim = _calc_sim(regions[i], regions[j], img.shape[0] * img.shape[1])
            similarities.append((sim, i, j))
    
    # Merge regions
    regions = _merge_regions(regions, similarities, img.shape[0] * img.shape[1])
    
    # Convert regions to bounding boxes
    rects = []
    for region in regions:
        rect = {
            'rect': (
                region["min_x"],
                region["min_y"],
                region["max_x"] - region["min_x"],
                region["max_y"] - region["min_y"]
            ),
            'size': region["size"],
            'labels': region["labels"]
        }
        rects.append(rect)
    
    return img_segments, rects
