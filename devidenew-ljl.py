"""
author:Jessie
data:2022-05-12
update:2022-10-14
"""
import argparse
from genericpath import isdir
import os
import shutil
import random
import numpy as np
from torch import rand, seed

# You need to change this path to your img dir
img_root_path = "/home/ljl/Data/REID/ORS_REID"

if not os.path.isdir(img_root_path):
    print("please change the root_path")

img_grouped_path = img_root_path + "/all/merge"

total_img_num = 0
total_img_index = []
for root, dirs, files in os.walk(img_grouped_path, topdown=True):
    for dir in dirs:
        total_img_num += 1
        total_img_index.append(int(dir))
print("total img num: ", total_img_num)

# --------------------------------
# Options
# --------------------------------
praser = argparse.ArgumentParser(
    description="divided into training set and testing set"
)
praser.add_argument(
    "--odd_even",
    action="store_true",
    help="wether to use odd_even mode to divide the data",
)
praser.add_argument("--ratio", default=0.8, type=float, help="the ratio of division")

opt = praser.parse_args()

# ratio mode
# python dividenew-ljl.py --ratio <your ratio>

# odd_even mode
# python dividenew-ljl.py --odd_even

if opt.odd_even:
    img_root_path_oe = img_root_path + "/ORS_REID_OE"
    # --------------------------------
    # 以单双号划分训练，测试集
    # --------------------------------
    # gallery（odd）--testing set
    gallery_path = img_root_path_oe + "/gallery"
    if not os.path.isdir(gallery_path):
        os.mkdir(gallery_path)
    target_root_path = gallery_path
    for root, dirs, files in os.walk(img_grouped_path, topdown=True):
        for dir in dirs:
            if (int(dir) % 2) != 0:
                src_path = os.path.join(root, dir)
                tar_path = target_root_path + "/" + dir
                if os.path.isdir(tar_path):
                    shutil.rmtree(tar_path)
                if not os.path.isdir(tar_path):
                    shutil.copytree(src_path, tar_path)

    # --------------------------------
    # train_all（even）--training set
    trainall_path = img_root_path_oe + "/train_all"
    if not os.path.isdir(trainall_path):
        os.mkdir(trainall_path)
    target_root_path = trainall_path
    for root, dirs, files in os.walk(img_grouped_path, topdown=True):
        for dir in dirs:
            if (int(dir) % 2) == 0:
                src_path = os.path.join(root, dir)
                tar_path = target_root_path + "/" + dir
                if os.path.isdir(tar_path):
                    shutil.rmtree(tar_path)
                if not os.path.isdir(tar_path):
                    shutil.copytree(src_path, tar_path)
else:
    img_root_path_dvd = img_root_path + "/ORS_REID_DVD"
    if not os.path.isdir(img_root_path_dvd):
        os.mkdir(img_root_path_dvd)
    # --------------------------------
    # 以给定比例划分训练，测试集
    # --------------------------------
    random.seed(1)

    train_ratio = opt.ratio
    train_num = round(train_ratio * total_img_num)
    train_index = random.sample(range(1, total_img_num), train_num)
    print("train_num: ", len(train_index))
    print("train_index: ", train_index)

    test_ratio = 1 - train_ratio
    test_num = total_img_num - train_num
    mask = np.in1d(total_img_index, train_index, invert=True)
    test_index = np.array(total_img_index)[mask]
    print("test_num: ", len(test_index))
    print("test_index: ", test_index)
    # --------------------------------
    # gallery --testing set
    gallery_path = img_root_path_dvd + "/gallery"
    if not os.path.isdir(gallery_path):
        os.mkdir(gallery_path)
    target_root_path = gallery_path
    for root, dirs, files in os.walk(img_grouped_path, topdown=True):
        for dir in dirs:
            if int(dir) in test_index:
                src_path = os.path.join(root, dir)
                tar_path = target_root_path + "/" + dir
                if os.path.isdir(tar_path):
                    shutil.rmtree(tar_path)
                if not os.path.isdir(tar_path):
                    shutil.copytree(src_path, tar_path)
    # --------------------------------
    # train_all --training set
    trainall_path = img_root_path_dvd + "/train_all"
    if not os.path.isdir(trainall_path):
        os.mkdir(trainall_path)
    target_root_path = trainall_path
    for root, dirs, files in os.walk(img_grouped_path, topdown=True):
        for dir in dirs:
            if int(dir) in train_index:
                src_path = os.path.join(root, dir)
                tar_path = target_root_path + "/" + dir
                if os.path.isdir(tar_path):
                    shutil.rmtree(tar_path)
                if not os.path.isdir(tar_path):
                    shutil.copytree(src_path, tar_path)

# --------------------------------
# 从gallery中分出query
if opt.odd_even:
    query_path = img_root_path_oe + "/query"
else:
    query_path = img_root_path_dvd + "/query"

if not os.path.isdir(query_path):
    os.mkdir(query_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for dir in dirs:
        for rootin, dirsin, filesin in os.walk(
            os.path.join(gallery_path, dir), topdown=True
        ):
            i = 0
            for name in filesin:
                if not name[-3:] == "png":
                    continue
                if i == 0:
                    ID = name.split("_")
                    src = os.path.join(rootin, name)
                    tar = query_path + "/" + ID[0]
                    if os.path.isdir(tar):
                        shutil.rmtree(tar)
                    if not os.path.isdir(tar):
                        os.mkdir(tar)
                    shutil.copyfile(src, tar + "/" + name)
                    i += 1

# --------------------------------
# 从train_all中分出train,val
if opt.odd_even:
    train_path = img_root_path_oe + "/train"
    val_path = img_root_path_oe + "/val"
else:
    train_path = img_root_path_dvd + "/train"
    val_path = img_root_path_dvd + "/val"

if not os.path.isdir(train_path):
    os.mkdir(train_path)
if not os.path.isdir(val_path):
    os.mkdir(val_path)

for root, dirs, files in os.walk(trainall_path, topdown=True):
    for dir in dirs:
        for rootin, dirsin, filesin in os.walk(
            os.path.join(trainall_path, dir), topdown=True
        ):
            for name in filesin:
                if not name[-3:] == "png":
                    continue
                ID = name.split("_")
                src = os.path.join(rootin, name)
                tar = train_path + "/" + ID[0]
                if not os.path.isdir(tar):
                    os.mkdir(tar)
                    tar = val_path + "/" + ID[0]
                    os.mkdir(tar)
                shutil.copyfile(src, tar + "/" + name)
