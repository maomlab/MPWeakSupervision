import os
import platform
import time
import argparse
import multiprocessing
import functools
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split

from utils import *
from dataset import *
from modules import SmallDeepSet, profile_AttSet, simpling_pooling, transformer


import wandb


def set_args():
    # Training settings
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to training data parquet file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to training data parquet file.",
    )
    parser.add_argument(
        "--feature_columns_path", type=str)
    parser.add_argument("--response_column", type=str, default="time_point")

    parser.add_argument(
        "--id_columns",
        type=str,
        nargs="+",
        default=["Plate_Name", "row", "column"],
        help="List of columns used to identify which cells go to which split")
    
    ##parser.add_argument("--train_idx", type=str, default="../../raw_data/weak_supervision/wellID_train_idx.txt")
    ##parser.add_argument("--test_idx", type=str, default="../../raw_data/weak_supervision/wellID_test_idx.txt")
    #parser.add_argument('--normalize', action='store_true', help="Standardize test and train features separately")
    #parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    #parser.set_defaults(normalize=True)
    # model
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--model_paths", type=str, default="")

    parser.add_argument("--model_name", type=str, default="SmallDeepSet")
    parser.add_argument("--pool_method", type=str, default="mean")
    parser.add_argument("--prediction_mode", type=str, default="regression")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training.")
    parser.add_argument("--no-cuda", action="store_false", dest="cuda")
    parser.set_defaults(cuda=True)

    # training
    parser.add_argument("--cpu_count", type=int, default=1, help="number of cpus to use for data loading")
    parser.add_argument("--device", type=int, default=0, help="device for cuda")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--bag_size", type=int, default=100)
    parser.add_argument("--bag_group_by", type=str, nargs="+", default=["Plate_Name", "row", "column"])
    parser.add_argument(
        "--resample_bags_frequency", type=int, default=5,
        help = "Resample the train and test bags after this many epochs")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate_scheduler", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # artifacts
    parser.add_argument("--wandb_entity", type=str, default="maomlab")
    parser.add_argument("--wandb_project", type=str, default="SARS-CoV-2_TimeSeries")
    parser.add_argument("--output_dir", type=str, default="../../intermediate_data/weak_supervision/")

    args = parser.parse_args()

    return args

def check_args(args):
    if not os.path.exists(args.feature_columns_path):
        raise Exception(f"Path to feature columns file '{args.feature_columns_path}' doesn't exist. It should be a tab seprated table with columns ['feature', 'transform'].")
    
    if args.cuda:
        if not torch.cuda.is_available():
            print("WARNING: CUDA training is requested but it is not available")
            args.cuda = False
    else:
        if torch.cuda.is_available():
            print("WARNING: CUDA training is not requested but it is available")

    valid_model_names = ["SmallDeepSet", "simple_pooling", "profile_AttSet", "transformer"]
    if args.model_name not in valid_model_names:
        raise Exception(f"Unrecognized model_name '{args.model_name}', valid model names are [{','.join(valid_model_names)}]")
    elif args.model_name == "profile_AttSet" and args.pool_method != "att":
        raise Exception(f"The only available pooling method for 'profile_AttSet' is 'att', but you supplied {args.pool_method}")

    valid_optimizer_names = ["Adam", "AdamW"]
    if args.optimizer not in valid_optimizer_names:
        raise Exception(f"Unrecognized optimizer '{args.optimizer}', valid optimizers are [{','.join(valid_optimizer_names)}]")

    valid_learning_rate_schedulers = [None, "ExponentialLR", "OneCycleLR", "CosineAnnealingWarmRestarts"]
    if args.learning_rate_scheduler not in valid_learning_rate_schedulers:
        raise Exception(f"Unrecognized learning rate scheduler '{args.learning_rate_scheduler}', valid learning rate schedulers are [{','.join(valid_learning_rate_schedulers)}]")
    
    return args


def setup_wandb(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_plate_data(data_path, id_columns, feature_columns, response_column, normalize):
    print(f"loading data {data_path} ...")
    plate_data = pa.parquet.read_table(
        source=data_path,
        columns = id_columns + feature_columns['feature'].tolist() + [response_column]).to_pandas()
    print(f"  Loaded data with shape {plate_data.shape} for plate {data_path}")
    if normalize:
        print(f"  normalizing features {data_path} ...")
        plate_data = transform_features(plate_data, feature_columns)
        print(f"  standardizing features {data_path} ...")
        plate_data = standardize_features(plate_data, feature_columns = feature_columns['feature'])
    print(f"  recoding response column for plate {data_path} ...")
    return plate_data


    
def main():
    args = set_args()
    args = check_args(args)
    
    if args.seed is not None:
        seed_everything(args.seed)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu = platform.processor()
    gpu = torch.cuda.get_device_name(DEVICE)

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"CPU type: {cpu}")
    print(f"GPU type: {gpu}")

        
    setup_wandb(args)


    #################
    ### Load data ###
    #################

    with open(args.feature_columns_path) as f:
        feature_columns = pd.read_csv(args.feature_columns_path, sep = "\t")

    print("loading training metadata ...")
    meta_train = pa.parquet.read_table(
        source = args.train_data_path,
        columns = args.id_columns).to_pandas()
    print("loading training features ...")    
    X_train = pa.parquet.read_table(
        source=args.train_data_path,
        columns = feature_columns['feature'].tolist()).to_pandas()
    print("loadingtraining response ...")        
    y_train = pa.parquet.read_table(
        source=args.train_data_path,
        columns = [args.response_column]).to_pandas()
    y_train = y_train[args.response_column]

    print("loading testing metadata ...")            
    meta_test = pa.parquet.read_table(
        source = args.test_data_path,
        columns = args.id_columns).to_pandas()
    print("loading testing features ...")    
    X_test = pa.parquet.read_table(
        source=args.test_data_path,
        columns = feature_columns['feature'].tolist()).to_pandas()
    print("loading testing response ...")                    
    y_test = pa.parquet.read_table(
        source=args.test_data_path,
        columns = [args.response_column]).to_pandas()
    y_test = y_test[args.response_column]

    print(f"loaded training data with shape {X_train.shape} and test data with shape {X_test.shape}")
    
    #if args.cpu_count == 1:
    #    data = []
    #    for data_path in args.data_path:
    #        data.append(load_plate_data(
    #            data_path = data_path,
    #            id_columns = args.id_columns,
    #            feature_columns = feature_columns,
    #            response_column = args.response_column,
    #            normalize = args.normalize))
    #else:
    #    map_fn = functools.partial(
    #        load_plate_data,
    #        id_columns = args.id_columns,
    #        feature_columns =  feature_columns,
    #        response_column = args.response_column,
    #        normalize = args.normalize)
    #    with multiprocessing.Pool(args.cpu_count) as p:
    #        data = p.map(map_fn, iter(args.data_path))
    #data = pd.concat(data, ignore_index=True)

    
    #blacklist = [
    #    "schema",
    #    "dose_nM",
    #    "Nuclei_Distance_Centroid_InfectedCells",
    #    "_Parent_",
    #    "Location",
    #    "AreaShape",
    #    "_Number_Object_Number",
    #    "_Count",
    #    "Positive",
    #    "Negative",
    #    "row",
    #    "column",
    #    "Plate_Name",
    #    "Condition",
    #    "Compound",
    #    "master_plate_id",
    #    "plate_id",
    #    "is_control",
    #    "time_point",
    #    "Metadata",
    #    "Image_Count_InfectedCells",
    #    "ImageNumber",
    #]

    #feature_list = []
    #for i in data.columns:
    #    if not any([j in i for j in blacklist]):
    #        feature_list.append(i)
    #
    # n_input_features = len(feature_list)

    #data[args.response_column] = (data[args.response_column].str.split(" ", expand=True)[0].astype(int))
    

    #splits = pd.read_csv(args.splits_path, sep="\t")
    #train_split = splits[splits["split"].isin(args.train_split)][args.id_columns]
    #train_split.set_index(args.id_columns, inplace=True)
    #data.set_index(args.id_columns, inplace=True)
    #train = data.join(train_split)
    #X_train = train[feature_columns['feature']]
    #
    #y_train = train[args.response_column]
    #
    #test_split = splits[splits["split"].isin(args.test_split)][args.id_columns]
    #test_split.set_index(args.id_columns, inplace=True)
    #test = data.join(test_split)
    #X_test = test[feature_columns['feature']]
    #y_test = test[args.response_column]
    
    #with open(args.train_idx, "r") as f:
    #    wellID_train_idx = f.read().splitlines()
    #
    #with open(args.test_idx, "r") as f:
    #    wellID_test_idx = f.read().splitlines()
    #
    #train = data[
    #    data[args.well_column].map(lambda x: x in wellID_train_idx)
    #]
    #test = data[
    #    data[args.well_column].map(lambda x: x in wellID_test_idx)
    #]
    #
    #if args.normalize:
    #    X_train = data_standardization(train[feature_list])
    #    X_val = data_standardization(test[feature_list])
    #else:
    #    X_train = train[feature_list]
    #    X_val = test[feature_list]
    #    
    #
    #y_train = train[args.response_column]
    #y_val = test[args.response_column]

    # define model
    if args.model_name == "SmallDeepset":
        model = SmallDeepSet(704, pool=args.pool_method, reg=args.prediction_mode == "regression")
    elif args.model_name == "simple_pooling":
        model = simple_pooling(704, pool=args.pool_method, reg=args.prediction_mode == "regression")
    elif args.model_name == "profile_AttSet":
        model = profile_AttSet(704, pool=args.pool_method, reg=args.prediction_mode == "regression")
    elif args.model_name == "transformer":
        model = transformer(704, reg=args.prediction_mode == "regression")        
    else:
        raise Exception(f"Unrecognized model_name '{args.model_name}'")

    if args.pretrained:
        model.load_state_dict(
            torch.load(args.model_paths, map_location=torch.device(args.device))
        )

    if args.cuda:
        model.cuda(args.device)

    # define opt
    if args.optimizer == "Adam":
        opt = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
        )
    elif args.optimizer == "AdamW":
        opt = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
        )
    else:
        raise Exception(f"Unrecognized optimizer '{args.optimizer}'")

    if args.learning_rate_scheduler == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer = opt,
            max_lr = args.max_lr,
            total_steps = args.epochs * 1000)
    elif args.learning_rate_scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma = 0.9)
    elif args.learning_rate_scheduler == "CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = opt, T_0 = 10)
    else:
        scheduler = None

    # start training
    current_smooth = 100000000
    for epoch in range(args.epochs):
        epoch_result = []
        if epoch % args.resample_bags_frequency == 0:
            train_bag_data = TimeSeriesProfileBag(
                df=X_train,
                y=y_train,
                sample_idx=make_bags(
                    data=meta_train,
                    bag_size=args.bag_size,
                    group_by=args.bag_group_by,
                ),
                cuda=args.cuda,
                gpu=args.device,
            )
            test_bag_data = TimeSeriesProfileBag(
                df=X_test,
                y=y_test,
                sample_idx=make_bags(
                    data=meta_test,
                    bag_size=500,
                    group_by=args.bag_group_by,
                    sample=False,
                ),
                cuda=args.cuda,
                gpu=args.device,
            )
            train_loader = D.DataLoader(
                train_bag_data, batch_size=args.batch_size, shuffle=True
            )
            test_loader = D.DataLoader(test_bag_data, batch_size=1, shuffle=True)

        valid_smooth, valid_mse = regression_train(
            epoch, train_loader, test_loader, model, opt, scheduler, args
        )

        if epoch > 16 == 0 and valid_smooth < current_smooth:
            current_smooth = valid_smooth
            torch.save(
                model.state_dict(),
                os.path(args.output_dir,  model_name + ".pth"))
            print("saved trained model")

        wandb.watch(model)


def regression_train(epoch, loader, val_loader, model, opt, scheduler, args):
    t = time.time()
    model.train()
    smooth_all = []
    mse_all = []
    for batch_idx, (data, bag_label) in enumerate(loader):
        bag_label = bag_label[:, 0].unsqueeze(1)
        if args.cuda:
            data, bag_label = data.cuda(args.device), bag_label.cuda(args.device)
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        opt.zero_grad()
        # calculate loss and metrics
        y_prob = model(data)
        loss = nn.SmoothL1Loss()(y_prob, bag_label)
        if (batch_idx % 50 == 0):
            print(f"loss batch index {batch_idx}: {loss.detach().cpu().item()}")
        # backward pass
        loss.backward()
        # step
        opt.step()
        if args.learning_rate_scheduler is not None:
            scheduler.step()

        smooth_all.append(loss.detach().cpu().item())
        mse_all.append(nn.MSELoss()(y_prob, bag_label).detach().cpu().item())

    smoothVl, mseVl = validation(model, val_loader, args)

    if args.learning_rate_scheduler is not None:
        learning_rate = scheduler.get_last_lr()[0]
    else:
        learning_rate = args.lr

    wandb.log({
        "epoch" : epoch + 1,
        "learning_rate": learning_rate,
        "train_loss_mean" : np.mean(smooth_all),
        "train_loss_std" : np.std(smooth_all),
        "train_mse_mean" : np.mean(mse_all),
        "train_mse_std" : np.std(mse_all),
        "validation_loss_mean" : np.mean(smoothVl),
        "validation_loss_std" : np.std(smoothVl),
        "validation_mse_mean" : np.mean(mseVl),
        "validation_mse_std" : np.std(mseVl),
        "epoch_time_elapsed" : time.time() - t})

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

    return smooth_all, mse_all


if __name__ == "__main__":
    main()
