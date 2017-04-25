#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 22:48:30 2017

@author: zhaolei
"""
import mxnet as mx

from ssd_net import get_feature_layer,residual_predict,deconv_layer

from common import multibox_layer,conv_act_layer

feature_net1,feature_net2,feature_net3,feature_net4=get_feature_layer()

conv1, relu1 = conv_act_layer(feature_net4, "8_1", 512, kernel=(1,1), pad=(0,0), 
                              stride=(1,1), act_type="relu", use_batchnorm=False)
conv2, relu2 = conv_act_layer(relu1, "8_2", 512, kernel=(3,3), pad=(1,1), 
                              stride=(2,2), act_type="relu", use_batchnorm=False)
conv3, relu3 = conv_act_layer(relu2, "9_1", 512, kernel=(1,1), pad=(0,0), 
                              stride=(1,1), act_type="relu", use_batchnorm=False)

mx.viz.plot_network(relu1,shape={'data':(18,3,300,300)}).view() 
    
    