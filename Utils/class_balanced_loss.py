#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
Code from:
https://raw.githubusercontent.com/vandit15/Class-balanced-loss-pytorch/master/class_balanced_loss.py

Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples"
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger



def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class BalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma, device, label_smoothing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.label_smoothing=label_smoothing
    def forward(self, logits, labels):
        return CB_loss(labels,logits, self.samples_per_cls, self.no_of_classes, self.loss_type, self.beta, self.gamma, self.device, self.label_smoothing)

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device, label_smoothing):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    logger.debug(f'weights: {weights.shape}')
    weights = weights / np.sum(weights) * no_of_classes

    # labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda(device) # Federico changed here
    labels_one_hot = labels

    logger.debug(f'weights: {weights.shape}')
    # logger.debug(f'weights: {weights}')
    weights = torch.tensor(weights).float().cuda(device)
    weights_for_ce = weights.clone()
    logger.debug(f'weights: {weights.shape}')
    weights = weights.unsqueeze(0)
    logger.debug(f'weights: {weights.shape}')
    print(labels_one_hot.shape)
    print(weights.repeat(labels_one_hot.shape[0],1).shape)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    logger.debug(f'weights: {weights.shape}')
    weights = weights.sum(1)
    logger.debug(f'weights: {weights.shape}')
    weights = weights.unsqueeze(1)
    logger.debug(f'weights: {weights.shape}')
    weights = weights.repeat(1,no_of_classes)
    logger.debug(f'weights: {weights.shape}')
    # logger.debug(f'weights: {weights[0,:]}')

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "cross-entropy":
        cb_loss = F.cross_entropy(input=logits, target = labels_one_hot, weight = weights_for_ce, label_smoothing=label_smoothing)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


def test():
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print(cb_loss)