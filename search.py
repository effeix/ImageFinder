# Standard Library
import pickle
import sys

from os import listdir

# Third Party
import cv2
import matplotlib.pyplot as plt
import numpy as np

from random import sample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


N_CLUSTERS = 300

def descriptors(image):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, descs = surf.detectAndCompute(image, None)
    
    return kp, descs


def bulk_descriptors(directories, max_items=5):
    base_path = "./101_ObjectCategories/"
    img_list = []
    descs = []
    
    for d in directories:
        images = [base_path + d + "/" + i for i in listdir(base_path + d)[:max_items]]  
        img_list.extend(images)
        
        for img_path in images:
            img = cv2.imread(img_path, 0)
            
            kp, desc = descriptors(img)
            descs.append([img_path, desc])
            
    return img_list, np.array(descs, dtype=object)


def create_vocabulary(imgs, sz=N_CLUSTERS):
    dataset = []
    
    for img in imgs:
        for desc in img[1]:
            dataset.append(desc)
    
    # Normalize descriptors dataset for better results
    std_scaler = StandardScaler()
    scaled_dataset = std_scaler.fit_transform(dataset)
    kmeans = KMeans(n_clusters=sz).fit(scaled_dataset)
    
    # Return kmeans calculated groups.
    # Also return the scaler, because we need to use it in others functions
    # and it has already been trained with the dataset
    return (kmeans.cluster_centers_, kmeans, std_scaler)


def create_histogram(img, vocab):
    # to create the histogram, we need the frequency of each cluster
    
    centers = vocab[0]
    kmeans = vocab[1]
    std_scaler = vocab[2]
    
    scaled_img = std_scaler.transform(img)
    
    # predict() returns an array where each element is the index
    # of a cluster to which an instance in scaled_img belongs to:
    # [2, 45, 67, 3, 128, 204, 32, 1, ...]
    # 
    # this way, we don't need to manually calculate the distances to each centroid
    prediction = kmeans.predict(scaled_img)
    
    # histogram data: number of bins = number of clusters
    hist = np.ones(kmeans.n_clusters)
    
    # count how many times each cluster was matched
    for p in prediction:
        hist[p] += 1
        
    return hist


def create_histograms(imgs, vocab):
    hists = []
    for img in imgs:
        hist = create_histogram(img[1], vocab)
        hists.append([img[0], hist])
        
    return np.array(hists, dtype=object)

def compare_hist_chisqd(o, e):
    return np.sum((o - e)**2 / e)

def search(directory, all_hists, vocab, search_path):
    base_path = "./101_ObjectCategories/"
    
    distances = []

    # search_path = base_path + directory + "/" + f"image_00{img_path_idx}.jpg"
    img = cv2.imread(search_path, 0)
    kp, des = descriptors(img)

    search_hist = create_histogram(des, vocab)

    for h in db_hists:
        distances.append([h[0], compare_hist_chisqd(search_hist, h[1])])
    
    distances.sort(key=lambda x: x[1])
    return (search_path, distances[:5])

if __name__ == '__main__':
    
    # Train
    directories = [
        "Faces",
        "garfield",
        "platypus",
        "nautilus",
        "elephant",
        "gerenuk"
    ]

    image_paths, images = bulk_descriptors(directories, max_items=15)
    vocabulary = create_vocabulary(images)
    db_hists = create_histograms(images, vocabulary)

    # Search
    distances = []
    
    s = search(d, db_hists, vocabulary, sys.argv[1])
    distances.append(s)

    print(distances)
