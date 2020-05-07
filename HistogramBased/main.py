#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:56:16 2020

@author: Purnendu Mishra
"""

# Standard library import
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.io import imread

from mpl_toolkits import mplot3d
#%%
# Separate skin and non-skin data

root           = Path.home()/'Documents'/'MyCodes'/'SkinSegmentation'/'PrepareDataset'

skin_data_path = root/'skin_data.csv'
non_skin_path  = root/'non_skin_data.csv'

skin_df        = pd.read_csv(skin_data_path, header= None)
non_skin_df    = pd.read_csv(non_skin_path, header= None)

# def separate_data_classwise(dataframe = None):
#     classwise_data = {}

#     for u in dataframe.iloc[:,-1].unique():
#         samples = dataframe[dataframe[3] == u].values
#         classwise_data[u]  = np.array(samples[:,:-1], dtype = np.float32)

#     return classwise_data


def Get_pixel_probability(pixel = None, hist = None, bins_edge = None):
    x = pixel[0]
    y = pixel[1]
    z = pixel[2]

    u = np.digitize(x, bins_edge[0], right=True) - 1
    v = np.digitize(y, bins_edge[1], right=True) - 1
    w = np.digitize(z, bins_edge[2], right=True) - 1

    # print(u,v,w)

    return hist[u,v,w]


def Predict(image          = None,
            skin_hist      = None,
            non_skin_hist  = None,
            skin_bins      = None,
            non_skin_bins  = None,
            threshold      = 0.04):

    img  = image.reshape(-1,3)
    h, w = image.shape[:2]

    l    = len(img)

    mask = np.zeros((l,), dtype = np.float32)
    th   = threshold

    for i in range(l):
        pixel      = img[i]

        skin_prob      = Get_pixel_probability(pixel = pixel, hist = skin_hist, bins_edge = skin_edges)

        non_skin_prob  = Get_pixel_probability(pixel = pixel,
                                                hist  = non_skin_hist,
                                                bins_edge = non_skin_edges)


        ratio = (skin_prob + 1e-16) /(non_skin_prob + 1e-10)

        if ratio >= th:
            mask[i] = 1




    mask = mask.reshape((h,w))

    return mask





# # class_wise_data = separate_data_classwise(dataframe = df)

# # Create 3D histogram
skin_data     = skin_df.values
non_skin_data = non_skin_df.values


channels  = [0,1,2]
bins      = [32] * 3
ranges    = [[0,255],[0,255],[0,255]]

# Skin histogram
bs        = skin_data[:,2]
gs        = skin_data[:,1]
rs        = skin_data[:,0]

skin_hist, skin_edges = np.histogramdd((rs,gs,bs), bins, density= True, range=ranges)

# Non-skin histogram
b_ns     = non_skin_data[:,2]
g_ns     = non_skin_data[:,1]
r_ns     = non_skin_data[:,0]

non_skin_hist, non_skin_edges = np.histogramdd((r_ns, g_ns, b_ns), bins, density= True, range=ranges)
# # find probability of of each bin


#%% Test
test_image = 'test_01.jpg'

image      = imread(test_image)
image      = cv2.resize(image, (224, 224))

mask       = Predict(image           = image,
                     skin_hist       = skin_hist,
                     non_skin_hist   = non_skin_hist,
                     skin_bins       = skin_edges,
                     non_skin_bins   = non_skin_edges,
                     threshold       = 0.2)


# fig = plt.figure(figsize=(10,10))
# ax1 = plt.axes(projection="3d")
# ax1.scatter3D(skin_hist[:,:,0], skin_hist[:,:,1], skin_hist[:,:,2])
# ax1.scatter3D(non_skin_hist[:,:,0], non_skin_hist[:,:,1], non_skin_hist[:,:,2], c = 'g')
# plt.show()

fig = plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(mask)
plt.show()