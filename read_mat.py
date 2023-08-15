"""
author:Jessie
date:2022-10-06
founction:read .mat file
"""

import torch
import scipy.io as io

result = io.loadmat("./pytorch_result_1206.mat")
print(result.keys())

query_feature = torch.FloatTensor(result["query_f"])
print(query_feature.shape)
query_cam = result["query_cam"][0]
print(query_cam)
query_label = result["query_label"][0]
print(query_label)
gallery_feature = torch.FloatTensor(result["gallery_f"])
print(gallery_feature.shape)
gallery_cam = result["gallery_cam"][0]
print(gallery_cam)
gallery_label = result["gallery_label"][0]
print(gallery_label)
