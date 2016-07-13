#-*- coding: utf-8 -*-
__author__ = 'Administrator'
"""
使用theano实现降噪自编码
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
from logistic_sgd import load_data
from utils import tile_raster_images
import PIL.Image

class dA(object):
    """
    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)
    """

#定义测试DA算法的主函数
#input:
#       learning_rate: float 学习率    training_epochs: int 训练的代数
#       dataset: str 数据集    batch_size: int 块大小     output_folder: str 输出图片的文件夹
def test_dA(learning_rate=0.1,training_epochs=15,
            dataset='../data/mnist.pkl.gz',batch_size=20,
            output_folder='dA_plots'):
    #--------------初始化数据---------------
    datasets=load_data(dataset) #加载训练数据集
    train_set_x,train_set_y=datasets[0] #得到训练数据集、训练数据标签
    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size#得到训练块的大小
    #定义符号变量
    index=T.lscalar() #块内数据的索引
    x=T.matrix('x') #栅格化图像数据
    #定义输出图像的路径
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    #--------------------------------------
    #--------------建立无损模型------------



if __name__=="__main__":
    test_dA()