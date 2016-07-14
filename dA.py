#-*- coding:utf-8 -*-
#时间: 2016.7.13
#Email: zhaoyuafeu@gmail.com
'''
本教程介绍了使用theano实现降噪自编码的过程。

降噪自编码是堆栈自编码的组成模块。它是Bengio et al在2007年提出的。
假设自编码器的输入为x，将它映射到隐含层，y = f_{\theta}(x) = s(Wx+b)
其中变量为\theta={W,b}。 隐层输出y映射到重构矢量z \in [0,1]^d: z = g_{\theta'}(y) = s(W'y + b')
权重矩阵 W' 表示为W' = W^T, 所以自编码有约束权重。网络通过最小化重构误差(x和z的误差)来进行训练。

对于降噪自编码，在训练时，首先将x污染为 \tilde{x}，这里 \tilde{x}是通过随机映射而部分污染的x。
然后，y的计算同自编码一样(使用 \tilde{x}): y = s(W\tilde{x} + b) 同时 z 可以表示为 s(W'y + b').
重构误差是指z与未污染的x的误差，可以用交叉熵来表示：- \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

参考文献：
    - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
'''

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images
import PIL.Image