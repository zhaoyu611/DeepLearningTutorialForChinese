# -*- coding: utf-8 -*-
'''
时间：2016.8.4
作者：赵雨
E-mail: zhaoyuafeu@gmail.com
说明：针对deepleanring.net中SdA部分的翻译
'''
#######################################
"""
本教程使用Theano进行栈式自编码(SdA)。
SdA的基础是自编码器，该理论是Bengio等人在2007年提出的。
自编码器输入为x，并映射到隐含层 y = f_{\theta}(x) = s(Wx+b)
其中参数是\theta={W,b}。然后将隐层输出y映射输出重构向量z\in [0,1]^d
映射函数为z = g_{\theta'}(y) = s(W'y + b')。权重矩阵 W'可以由W' = W^T
得到，W'和W称为tied weights。网络的训练目标是最小化重构误差(x和z之间的误差)。

对于降噪自编码的训练，首先将x corrupted为\tilde{x}，\tilde{x}是x的破损形式，
破损函数是随机映射。随后采用与之前相同方法计算y(y = s(W\tilde{x} + b)
并且 z = s(W'y + b') )。重构误差是计算z和uncorrupted x的误差，即：
计算两者的交叉熵：
     - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

 参考文献：
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""

import cPickle
import gzip
import os
import sys
import time
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA

class SdA(object):
    """栈式自编码类(SdA)
    栈式自编码模型是由若干dAs堆栈组成。第i层的dA的隐层变成第i+1层的输入。
    第一层dA的输入是SdA的输入，最后一层dA的输出的SdA的输出、预训练后，
    SdA的运行类似普通的MLP，dAs只是用来初始化权重。
    """
    def __init__(self,numpy_rng,theano_rng=None,n_ins=784,
                 hidden_layers_sizes=[500,500],n_out=10,
                 corruption_levels=[0.1,0.1]):
        """
        该类可以构造可变层数的网络
        numpy_rng：numpy.random.RandomState  用于初始化权重的随机数

        theano_rng: theano.tensor.shared_randomstreams.RandomStreams
                    Theano随机生成数,如果，默认值为None， 则是由'rng'
                    生成的随机种子
        n_ins: int  SdA输入的维度
        hidden_layers_sizes: lists of ints 中间层的层数列表，最少一个元素
        n_out: int 网路输出量的维度
        corruption_levels: list of float 每一层的corruption level
        """

        self.sigmoid_layers=[]
        self.dA_layers=[]
        self.params=[]
        self.n_layers=len(hidden_layers_sizes)

        assert self.n_layers>0 #设定隐层数量大于0

        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2**30))

        #设定符号变量
        self.x=T.matrix('x') #栅格化的图像数据
        self.y=T.matrix('y') #由[int]型标签组成的一维向量

        #SdA是一个MLP，降噪自编码器共享中间层的权重向量。
        #首先将SdA构造为深层多感知器，然后构造每个sigmoid层。
        #同时，该层的降噪自编码器也会共享权重。
        #预训练过程是训练这些自编码器(同时也会改变多感知器的权重)
        #在微调过程，通过在MLP上采用随机梯度下降法完成SdA的训练

        #构造sigmoid层
        for i in xrange(self.n_layers):
            #输入量的大小是下层隐层单元数量（本层不是第一层）
            #输入量的大小是输入量的大小(本层是第一层)
            if i==0:
                input_size=n_ins
            else:
                input_size=hidden_layers_sizes[i-1]

            #本层的输入是下层隐层的激活(本层不是第一层);
            #本层的输入是SdA的输入(本层是第一层)
            if i==0:
                layer_input=self.x
            else:
                layer_input=self.sigmoid_layers[-1].output

            #定义sigmoid层
            sigmoid_layer=HiddenLayer(rng=numpy_rng,
                                      input=layer_input,
                                      n_in=input_size,
                                      n_out=hidden_layers_sizes[i],
                                      activation=T.nnet.sigmoid)
            #将sigmoid层添加到层列表
            self.sigmoid_layers.append(sigmoid_layer)
            #这是个哲学问题...
            #但是我们只想说sigmoid_layers的参数就是
            #SdA的参数
            #dA中可视偏置是dA的参数，而不是SdA的参数
            self.params.extend(sigmoid_layer.params)

            #构造降噪自编码器与该层共享权重
            dA_layer=dA(numpy_rng=numpy_rng,
                        theano_rng=theano_rng,
                        )

def test_SdA(finetune_lr=0.1,pretraining_epochs=15,
             pretrain_lr=0.001,training_epichs=1000,
             dataset='../data/mnist.pkl.gz',bath_size=1):
    pass

if __name__=='__main__':
    test_SdA()


