#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:10:58 2017

@author: sy
"""


import pydensecrf.densecrf as dense_crf
from cv2 import imread
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials
import cv2
import os
import numpy as np

# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.7)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=80,
    schan=13,
    compatibility=4,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

gaussian_pairwise = potentials.GaussianPotential(
    sigma=3, 
    compatibility=2,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

# =============================================================================
# Create CRF model and add potentials
# =============================================================================
#zero_unsure:  whether zero is a class, if its False, it means zero canb be any of other classes
# =============================================================================
# crf = crf_model.DenseCRF(
#     num_classes = 3,
#     zero_unsure = True,              # The number of output classes
#     unary_potential=unary,
#     pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
#     use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
# )
# =============================================================================
crf = crf_model.DenseCRF(
    num_classes =5,
    zero_unsure = False,              # The number of output classes
    unary_potential=unary,
    pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
    use_2d = 'rgb-1d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)



def crfing(rootpath,image,probabilities,count):
    # =============================================================================
    # Set the CRF model
    # =============================================================================
    #label_source: whether label is from softmax, or other type of label.
    crf.set_image(
        image=image,
        probabilities=probabilities,
        colour_axis=-1,                  # The axis corresponding to colour in the image numpy shape
        class_axis=-1,                   # The axis corresponding to which class in the probabilities shape
        label_source = 'label'           # where the label come from, 'softmax' or 'label'
    )
    crf.perform_inference(10)  # The CRF model will restart run.
    new_mask10 = crf.segmentation_map
    print(crf.kl_divergence)
    cv2.imwrite(rootpath+str(count)+".png",new_mask10)
    print(count)


# =============================================================================
# Load image and probabilities
# =============================================================================


# name,imgnum=[],[]
# rootpath='./dataset/inference'
# for f in os.listdir(rootpath+'/test1result'):
#     name.append(f)
#     imgnum.append(f[:-9])
#     image_name=f[:-9]+'.jpg'
#     probabilities_name=f
#     image_name_path=rootpath+'/test1/'+image_name
#     probabilities_name_path=rootpath+'/test1result/'+probabilities_name
#     image = imread(image_name_path)
#     probabilities = imread(probabilities_name_path)
#
#     colors, labels = np.unique(probabilities, return_inverse=True)
#     HAS_UNK = (50 or 100 or 150) in colors
#     count = f[:-9]
#     if HAS_UNK:
#         crfing(rootpath,image, probabilities,count)
#     else:
#         cv2.imwrite(rootpath + '/720crf3/' + str(count) + ".png", probabilities)
#         print(count)
# print(len(name))








