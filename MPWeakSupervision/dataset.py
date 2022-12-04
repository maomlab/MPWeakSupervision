import time
import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.autograd import Variable


def transform_features(data: pd.DataFrame, transform_info) -> pd.DataFrame:
    for feature, transform_type in transform_info[["feature", "transform"]].agg(
        tuple, 1
    ):
        if transform_type == "identity":
            continue
        elif transform_type == "log":
            data[feature] = np.log(data[feature])
        elif transform_type == "log1p":
            data[feature] = np.log(data[feature] + 1)
        else:
            raise Exception(
                f"Unrecognized transform type '{transform_type}' for feature '{feature}'"
            )
    return data


def standardize_features(data: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    if feature_columns is None:
        return pd.DataFrame(
            StandardScaler().fit_transform(data), columns=data.columns, index=data.index
        )
    else:
        feature_column_indices = [list(data.columns).index(i) for i in feature_columns]
        data.values[:, feature_column_indices] = StandardScaler().fit_transform(
            data[feature_columns]
        )
        return data


def get_bag_cell_idx(image_time_cell, bag_size, level, no_sampling=False):
    # level can be "Image_Metadata_WellID", "ImageNumber"
    # bag_size as hyper-parameter
    # set threshold 50
    # sample with replacement
    # first divide dataset into different images
    image_index = image_time_cell[level].value_counts().index
    # Use a list to store the cell index
    bag_cell_idx = []
    # iterate all images and sampling fixed size with replacement
    for image_number in image_index:
        cells_tobe_sampled = image_time_cell[image_time_cell[level] == image_number]
        if len(cells_tobe_sampled) > bag_size:
            for i in range(len(cells_tobe_sampled) // bag_size):
                bag_cell_idx.append(
                    cells_tobe_sampled.sample(frac=1).index[
                        i * bag_size : (i + 1) * bag_size
                    ]
                )
        elif not no_sampling:
            bag_cell_idx.append(
                cells_tobe_sampled.sample(n=bag_size, replace=True).index
            )
            # sampling with replacement
        else:
            bag_cell_idx.append(cells_tobe_sampled.sample(frac=1, replace=False).index)
    return bag_cell_idx


def make_bags(data, bag_size, group_by=None, sample=True, cpu_count=1):
    """
    Make bags for training and evaluation of weak supervisation models

    Parameters:
    data (pandas.DataFrame): data with the shape of the input data and a column to group_by if given
    bag_size (int): construct bags of the given size. If the number of samples in the
       data doesn't divide evenly, then fill up the slots in the last bag by sampling.
    group_by (list): Construct bags by first stratifying by the variables in the group_by
       the contents of each bag only come from one level of the group_by variables.
       the inputs here are passed to the pandas.DataFrame.groupby function.
    sample (boolean): If the bag size is larger than the number of rows or number of rows
       in a group, and if 'sample=True' sample with replacement to fill the bag. If
       'sample=False', take one sample each

    @input pandas.DataFrame
    """

    def sample_bags(x):
        ids = x.index.to_list()

        if bag_size < len(ids):
            # to make even bags, fill the last bag with replacement
            ids = np.concatenate(
                [
                    ids,
                    np.random.choice(
                        a=ids,
                        size=-len(ids)
                        % bag_size,  # this is enough to fill the last bag
                        replace=False,
                    ),
                ]
            )

            return np.random.choice(
                a=ids, size=(len(ids) // bag_size, bag_size), replace=False
            ).tolist()
        elif sample:
            return np.random.choice(ids, size=(1, bag_size), replace=True).tolist()
        else:
            np.random.shuffle(ids)
            return [ids]

    time_begin = time.time()

    bags = None
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]
        if cpu_count == 1:
            bags = []
            for key, cells in data.groupby(by=group_by):
                bags.extend(sample_bags(cells))
        elif cpu_count > 1:
            with multiprocessing.Pool(cpu_count) as p:
                bags = p.map(
                    group_fn, [group for name, group in data.groupby(by=group_by)]
                )
        else:
            raise Exception(
                f"Unrecognized cpu_count {cpu_count}, it should be an integer >= 1"
            )
    else:
        bags = sample_bags(data)
    time_end = time.time()
    print(f"make_bags runtime: {round(time_end - time_begin, 2)}")
    return bags


# group stratification sampling in sklearn
# each batch of data, get more different images
class TimeSeriesProfileBag(D.Dataset):
    def __init__(self, df, y, sample_idx, cuda=True, gpu=0):
        self.df = df
        self.y = y
        self.sample_idx = sample_idx
        self.cuda = cuda
        self.gpu = gpu

    def __getitem__(self, idx):
        data = self.df.loc[self.sample_idx[idx]]
        y = self.y.loc[self.sample_idx[idx]]
        X = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        if self.cuda:
            X = X.cuda(self.gpu)
            y = y.cuda(self.gpu)
        return X, y

    def __len__(self):
        return len(self.sample_idx)


class TimeSeriesProfileFullBag(D.Dataset):
    def __init__(self, df, y, cuda=True, gpu=0):
        self.df = df
        self.y = y
        self.cuda = cuda
        self.gpu = gpu

    def __getitem__(self, idx):
        data = self.df.loc[self.sample_idx[idx]]
        y = self.y.loc[self.sample_idx[idx]]
        X = torch.tensor(data.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        if self.cuda:
            X = X.cuda(self.gpu)
            y = y.cuda(self.gpu)
        return X, y

    def __len__(self):
        return len(self.sample_idx)
