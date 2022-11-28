import numpy as np
from os.path import isfile
from os import rename
import scipy.sparse as sp
from pickle import load, dump
import gzip
import torch


def makeLogFile(filename="lossHistory.txt"):
    if isfile(filename):
        rename(filename, "lossHistoryOld.txt")

    with open(filename, "w") as text_file:
        print(
            "Epoch\tsmoothTr\tsmoTrStd\tmseTr\tmseTrStd\tsmoothVl\tsmoVlStd\tmseVl\tmseVlStd\ttime(s)",
            file=text_file,
        )
    print("Log file created...")
    return


def writeLog(
    logFile,
    epoch,
    lossTr,
    lTrStd,
    mseTr,
    mseTrStd,
    lossVl,
    lVlStd,
    mseVl,
    mseVlStd,
    eTime,
):
    print(
        "Epoch:{:04d}\t".format(epoch + 1),
        "smoothTr:{:.4f}\t".format(lossTr),
        "mseTr:{:.4f}\t".format(mseTr),
        "smoothVl:{:.4f}\t".format(lossVl),
        "mseVl:{:.4f}\t".format(mseVl),
        "time:{:.4f}".format(eTime),
    )

    with open(logFile, "a") as text_file:
        print(
            "{:04d}\t".format(epoch + 1),
            "{:.4f}\t".format(lossTr),
            "{:.4f}\t".format(lTrStd),
            "{:.4f}\t".format(mseTr),
            "{:.4f}\t".format(mseTrStd),
            "{:.4f}\t".format(lossVl),
            "{:.4f}\t".format(lVlStd),
            "{:.4f}\t".format(mseVl),
            "{:.4f}\t".format(mseVlStd),
            "{:.4f}".format(eTime),
            file=text_file,
        )
