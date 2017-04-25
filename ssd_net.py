#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:02:22 2017

@author: zhaolei
"""

import mxnet as mx

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, memonger=False):
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

def resnext(units, num_stages, filter_list, num_classes, num_group, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group, 
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            
        yield body


def deconv_layer(layer1,layer2,deconv_kernel=(3,3),deconv_pad=(1,1)):
    
    deconv=mx.sym.Deconvolution(data=layer1,num_filter=512,kernel=deconv_kernel,stride=(2,2),pad=deconv_pad)
    bn1=mx.sym.BatchNorm(data=deconv)
    act1=mx.sym.Activation(data=bn1,act_type='relu')
    
    conv1=mx.sym.Convolution(data=act1,num_filter=512,kernel=(3,3),pad=(1,1),stride=(1,1))
    bn2=mx.sym.BatchNorm(data=conv1)
    
    conv2=mx.sym.Convolution(data=layer2,num_filter=512,kernel=(3,3),pad=(1,1),stride=(1,1))
    bn3=mx.sym.BatchNorm(data=conv2)
    act2=mx.sym.Activation(data=bn3,act_type='relu')
    
    conv3=mx.sym.Convolution(data=act2,num_filter=512,kernel=(3,3),pad=(1,1),stride=(1,1))
    bn4=mx.sym.BatchNorm(data=conv3)
    
    product=bn4*bn2
    return mx.sym.Activation(data=product,act_type='relu')
    
def residual_predict(data):
    conv1=mx.sym.Convolution(data=data,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0))
    bn1=mx.sym.BatchNorm(data=conv1)
    act1=mx.sym.Activation(data=bn1,act_type='relu')
    
    conv2=mx.sym.Convolution(data=act1,num_filter=256,kernel=(1,1),stride=(1,1),pad=(0,0))
    bn2=mx.sym.BatchNorm(data=conv2)
    act2=mx.sym.Activation(data=bn2,act_type='relu')
    
    conv3=mx.sym.Convolution(data=act2,num_filter=1024,kernel=(1,1),stride=(1,1),pad=(0,0))
    bn3=mx.sym.BatchNorm(data=conv3)
    
    conv4=mx.sym.Convolution(data=data,num_filter=1024,kernel=(1,1),stride=(1,1),pad=(0,0))
    bn4=mx.sym.BatchNorm(data=conv4)
    short_cut= bn4+bn3
    return mx.sym.Activation(data=short_cut,act_type='relu')
    

def get_feature_layer(num_classes,conv_workspace=256,num_group=32):
    filter_list = [64, 128, 128, 256, 512]
    bottle_neck = True
    num_stages = 4
    units = [3, 4, 6, 3]
    return resnext(units      = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  num_group   = num_group, 
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)

net,net1,net2,net3 = get_feature_layer(10)
#ne2=residual_predict(net3)
#ne3=mx.sym.Convolution(data=ne2,num_filter=1024,kernel=(3,3),stride=(2,2),pad=(1,1))
#ne4=mx.sym.Deconvolution(data=ne3,num_filter=512,kernel=(2,2),stride=(2,2),pad=(0,0))
#ne=deconv_layer(ne3,ne2)
mx.viz.plot_network(net3,shape={'data':(128,3,300,300)}).view()   
