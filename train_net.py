#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 22:48:30 2017

@author: zhaolei
"""
import mxnet as mx

from ssd_net import get_feature_layer,residual_predict,deconv_layer

from common import multibox_layer,conv_act_layer

def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    
    label = mx.symbol.Variable(name="label")
    
    feature_net1,feature_net2,feature_net3,feature_net4=get_feature_layer()

    conv1, relu1 = conv_act_layer(feature_net4, "8_1", 512, kernel=(1,1), pad=(0,0), 
                                  stride=(1,1), act_type="relu", use_batchnorm=False)
    conv2, relu2 = conv_act_layer(relu1, "8_2", 512, kernel=(3,3), pad=(1,1), 
                                  stride=(2,2), act_type="relu", use_batchnorm=False)
    conv3, relu3 = conv_act_layer(relu2, "9_1", 512, kernel=(1,1), pad=(0,0), 
                                  stride=(1,1), act_type="relu", use_batchnorm=False)
    conv4, relu4 = conv_act_layer(relu3, "9_2", 256, kernel=(3,3), pad=(1,1), 
                                      stride=(2,2), act_type="relu", use_batchnorm=False)
    deconv1=deconv_layer(relu4,relu2)
    deconv2=deconv_layer(deconv1,feature_net4,deconv_kernel=(2,2),deconv_pad=(0,0))
    deconv3=deconv_layer(deconv2,feature_net3)
    layer1=residual_predict(relu4)
    layer2=residual_predict(deconv1)
    layer3=residual_predict(deconv2)
    layer4=residual_predict(deconv3)
    from_layers = [layer4, layer3, layer2, layer1]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
    normalizations = [20, -1, -1, -1]
    sizes = [0.1,0.8]
    num_channels = [512]
    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=False, interm_layer=0)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out

def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
net=get_symbol()
mx.viz.plot_network(net).view() 
    
    