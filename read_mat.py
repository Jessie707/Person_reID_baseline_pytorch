"""
author:Jessie
date:2022-10-06
founction:read .mat file
"""

import scipy.io as io

data = io.loadmat("./pytorch_result.mat")
print(data.keys())
print("query_label： ", data["query_label"])
print("query_cam： ", data["query_cam"])
