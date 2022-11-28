import os
import time
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *
from dataset import *
from modules import SmallDeepSet, profile_AttSet


def set_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--device", type=int, default=0, help="device for cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="covid19cq1_SARS_TS2PL1_Cell_MasterDataTable.parquet",
        help="Path to training data file.",
    )
    parser.add_argument("--train_idx", type=str, default="train_index.txt")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default="1")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    if args.seed is not None:
        seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    PL1_Cell = pa.parquet.read_table(source=args.data_path).to_pandas()
    print("loading data successfully")
    blacklist = [
        "schema",
        "dose_nM",
        "Nuclei_Distance_Centroid_InfectedCells",
        "_Parent_",
        "Location",
        "AreaShape",
        "_Number_Object_Number",
        "_Count",
        "Positive",
        "Negative",
        "row",
        "column",
        "Plate_Name",
        "Condition",
        "Compound",
        "master_plate_id",
        "plate_id",
        "is_control",
        "time_point",
        "Metadata",
        "Image_Count_InfectedCells",
        "ImageNumber",
    ]

    feature_list = []
    for i in PL1_Cell.columns:
        if not any([j in i for j in blacklist]):
            feature_list.append(i)

    PL1_Cell = PL1_Cell[PL1_Cell["time_point"] != "Uninfected"]
    PL1_Cell["time_point"] = (
        PL1_Cell["time_point"].str.split(" ", expand=True)[0].astype(int)
    )

    with open("train_index.txt", "r") as f:
        train_index = f.read().splitlines()
    train_index = [int(x) for x in train_index]

    with open("test_index.txt", "r") as f:
        test_index = f.read().splitlines()
    test_index = [int(x) for x in test_index]

    train, test = PL1_Cell.loc[train_index], PL1_Cell.loc[test_index]
    X_train, train_image_time_cell = (
        train[feature_list],
        train[["time_point", "ImageNumber"]],
    )
    X_val, test_image_time_cell = (
        test[feature_list],
        test[["time_point", "ImageNumber"]],
    )

    X_train = data_standardization(X_train)
    X_val = data_standardization(X_val)

    y_train = train_image_time_cell["time_point"]
    y_val = test_image_time_cell["time_point"]

    # define model
    model = profile_AttSet(612, reg=True)
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.model_paths, map_location=torch.device("cpu"))
        )
    if args.cuda:
        model.cuda(args.device)

    # define opt
    opt = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
    )

    model_name = time.strftime("logs/%m%d_%H_%M_%S") + "_mean"
    logFile = model_name + ".txt"
    makeLogFile(logFile)

    # start training
    current_smooth = 100000000
    for epoch in range(args.epochs):
        epoch_result = []
        if epoch % 5 == 0:
            train_bag_data = TimeSeriesProfileBag(
                X_train,
                y_train,
                get_bag_cell_idx(train_image_time_cell, 300),
                args.cuda,
                args.device,
            )
            test_bag_data = TimeSeriesProfileBag(
                X_val,
                y_val,
                get_bag_cell_idx(test_image_time_cell, 300),
                args.cuda,
                args.device,
            )
            train_loader = D.DataLoader(
                train_bag_data, batch_size=args.batch_size, shuffle=True
            )
            test_loader = D.DataLoader(
                test_bag_data, batch_size=args.batch_size, shuffle=True
            )

        valid_smooth, valid_mse = regression_train(
            epoch, train_loader, test_loader, model, opt, args, logFile
        )

        if epoch > 16 == 0 and valid_smooth < current_smooth:
            current_smooth = valid_smooth
            torch.save(model.state_dict(), model_name + ".pth")
            print("saved trained model")


def regression_train(epoch, loader, val_loader, model, opt, args, logFile):
    t = time.time()
    model.train()
    smooth_all = []
    mse_all = []
    for batch_idx, (data, bag_label) in enumerate(loader):
        bag_label = bag_label[:, 0].unsqueeze(1)
        if args.cuda:
            data, bag_label = data.cuda(args.device), bag_label.cuda(args.device)
        model.train()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        opt.zero_grad()
        # calculate loss and metrics
        y_prob = model(data)
        loss = nn.SmoothL1Loss()(y_prob, bag_label)
        # backward pass
        loss.backward()
        # step
        opt.step()

        smooth_all.append(loss.detach().cpu().item())
        mse_all.append(nn.MSELoss()(y_prob, bag_label).detach().cpu().item())

    mseVl, smoothVl = validation(model, val_loader, args)
    writeLog(
        logFile,
        epoch,
        np.mean(smooth_all),
        np.std(smooth_all),
        np.mean(mse_all),
        np.std(mse_all),
        np.mean(smoothVl),
        np.std(smoothVl),
        np.mean(mseVl),
        np.std(mseVl),
        time.time() - t,
    )

    return np.mean(smoothVl), np.mean(mseVl)


def validation(model, loader, args):
    smooth_all = []
    mse_all = []

    for batch_idx, (data, bag_label) in enumerate(loader):
        bag_label = bag_label[:, 0].unsqueeze(1)
        if args.cuda:
            model.cuda(args.device)
            data, bag_label = data.cuda(args.device), bag_label.cuda(args.device)
        model.eval()

        data, bag_label = Variable(data), Variable(bag_label)
        # calculate loss and metrics
        y_prob = model(data)
        smooth = nn.SmoothL1Loss()(y_prob, bag_label)
        smooth_all.append(smooth.detach().cpu())
        mse = nn.MSELoss()(y_prob, bag_label).detach().cpu()
        mse_all.append(mse)

    return np.mean(smooth_all), np.mean(mse_all)


if __name__ == "__main__":
    main()
