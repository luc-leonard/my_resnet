from typing import Tuple, List

from fastai.callback import mixup
from fastai.callback.mixup import  MixHandler
from fastai.callback.tracker import SaveModelCallback
from fastai.data.external import URLs
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastcore.foundation import L
from torch.nn import CrossEntropyLoss

from data import get_data
from model.resnet_with_stems import Resnet
import torch
import numpy as np
import torch.nn.functional as F

from train import fit


def get_learner(m, dls) -> Learner:
    learner = Learner(dls, m, loss_func=CrossEntropyLoss(), metrics=accuracy,
                      cbs=[
                          SaveModelCallback(fname='./saved_model_mixup_try'),
                      ])
    return learner.to_fp16()


def main():
    print('getting data')
    dls = get_data(URLs.IMAGENETTE_160, 320, 224)
    print('generating model')
    model = Resnet(dls.c, [3, 4, 6, 3], expansion=4)
    print(model)
    fit(5, model.to('cuda'), CrossEntropyLoss(), dls.train, dls.valid, lr=3e-3)
    # learner = get_learner(model, dls)
    # learner.fit_one_cycle(200, 3e-3)
    # learner.save()


if __name__ == '__main__':
    main()
