import argparse
import torch
import multiprocessing

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.kernels = 10

    opt.resources = "./"

    opt.size = 224

    opt.n_epochs = 200

    opt.description = opt.archname = 'ResNet50 Classification'
    opt.evaluate = False

    opt.lr = 0.01
    opt.step_size = 3

    opt.batch_size = 32
    opt.device = torch.device('cuda')

    return opt
