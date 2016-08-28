#-*-coding: utf-8 -*-
__author__ = 'Administrator'

"""
本代码使用Theano实现受限玻尔兹曼机(RBM)
玻尔兹曼机(BMs)是一种带隐藏变量的特殊形式的自由能模型。
受限玻尔兹曼机是不含可见层-可见层和隐层-隐层连接的形式。
"""
import cPickle
import gzip
import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
from logistic_sgd import load_data

class RBM(object):
    def __init__(self,input=None,n_visible=784,n_hidden=500,\
                 W=None,hbias=None,vbias=None,numpy_rng=None,
                 theano_rng=None):
        """
        RBM类定义了从隐层到可见层(反之亦然)的模型参数和基本操作，
        同时定义了CD的更新
        :param input:对于标准RBM，输入为None，对于大模型的RBM模块，输入为符号变量
        :param n_visible:可见单元的数量
        :param n_hidden:隐藏单元的数量
        :param W:对于标准RBM,输入为None；对于DBN网络的RBM模块，输入为共享权重矩阵的符号变量；
                在DBN中，RBMs和MLP的层共享权重。
        :param hbias:对于标准RBM，输入为None；对于大型网络的RBM模块，输入为符号变量共享
                隐层单元的偏置
        :param vbias:对于标准RBMs。输入为None；对于大型网络的RBM模块，
                输入为符号变量共享卡肩蹭单元的偏置
        :param numpy_rng: 生成的随机数
        :param theano_rng: 生成的符号变量的随机数
        :return:
        """
        #######################
        #####初始化模型参数#####
        #######################
        self.n_visible=n_visible
        self.n_hidden=n_hidden

        #生成随机数
        if numpy_rng is None:
            numpy_rng=numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng=RandomStreams(numpy_rng.randint(2**30))

        #W服从均匀分布，采样区间为-4*sqrt(n_hidden+n_visible)
        # 到-4*sqrt(n_hidden+n_visible)。如果将数据类型从
        #asarray转换到theano.config.floatX，那么程序可以在GPU运行
        if W is None:
            initial_W=numpy.asarray(numpy_rng.uniform(
                    low=-4*numpy.sqrt(n_hidden+n_visible),
                    high=4*numpy.sqrt(n_hidden+n_visible),
                    size=(n_visible,n_hidden)),
                    dtype=theano.config.floatX)
            #theano共享权重和偏置
            W=theano.shared(value=initial_W,name='W',borrow=True)

        #创建隐藏单元偏置的共享变量
        if hbias is None:
            hbias=theano.shared(value=numpy.zeros(n_hidden,
                                                  dtype=theano.config.floatX),
                                name='hbias',borrow=True)
        #创建可见单元偏置的共享变量
        if vbias is None:
            vbias=theano.shared(value=numpy.zeros(n_hidden,
                                                  dtype=theano.config.floatX),
                                name='hbias',borrow=True)
        #初始化标准RBM的输入层或DBN的layer0
        self.input=input
        if not input:
            self.input=T.matrix('input')

        self.W=W
        self.hbias=hbias
        self.vbias=vbias
        self.theano_rng=theano_rng
        #相比在函数中应用共享变量，将变量整合在一个列表中并不是好主意
        self.params=[self.W,self.hbias,self.vbias]

    #计算自由能
    def free_energy(self,v_sample):
        wx_b=T.dot(v_sample,self.W)+self.hbias
        vbias_term=T.dot(v_sample,self.vbias)
        hbias_term=T.sum(T.log(1+T.exp(wx_b)),axis=1)
        return -vbias_term-hbias_term

    #定义向上传播
    def propup(self,vis):
        '''
        定义从可见单元到隐藏单元的传播函数。注意函数的返回值是未sigmoid运算的值。
        As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        vis: 可见层单元
        '''
        pre_sigmoid_activation=T.dot(vis,self.W)+self.hbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]
    #给定v单元计算h单元的函数
    def sample_h_given_v(self,v0_sample):
        pre_sigmoid_h1,h1_mean=self.propup(v0_sample)
        #利用激活函数获得隐层样本
        #注意theano_rng.binomial返回dtype为int64的符号变量。
        #如果想在GPU上进行计算，需要将类型转换为floatX
        h1_sample=self.theano_rng.binomial(size=h1_mean.shape,
                                           n=1,p=h1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_h1,h1_mean,h1_sample]

    #定义向下传播
    def propdown(self,hid):
        '''
        定义从隐藏单元到可见单元的传播函数，注意函数的返回值是未sigmoid运算的值。
        As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        hid:隐层单元
        '''
        pre_sigmoid_activation=T.dot(hid,self.W.T)+self.vbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]


    #给定h单元计算v单元的函数
    def sample_v_given_h(self,h0_sample):
        pre_sigmoid_v1,v1_mean=self.propdown(h0_sample)
        v1_sample=self.theano_rng.binomial(size=v1_mean.shape,n=1,p=v1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_v1,v1_mean,v1_sample]

    #从隐藏状态出发，执行一步Gibbs采样过程
    def gibbs_hvh(self,h0_sample):
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_v_given_h(v1_sample)
        return [pre_sigmoid_v1,v1_mean,v1_sample,
                pre_sigmoid_h1,h1_mean,h1_sample]

    #从可见状态出发，执行一步Gibbs采样过程
    def gibbs_vhv(self,v0_sample):
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_h_given_v(h1_sample)
        return [pre_sigmoid_h1,h1_mean,h1_sample,
                pre_sigmoid_v1,v1_mean,v1_sample]

    def get_cost_updates(self,lr=0.1,persistent=None,k=1):
        """
        函数执行一步CD-k或者PCD-k
        :param lr: 训练RBM的学习率
        :param persistent: 对于CD,输入为None；对于PCD，输入为包含Gibbs链旧状态的
                           共享变量。它必须是size的共享变量(batch size，隐层单元数量)
        :param k: CD-k/PCD-k中Gibbs采样的步数
        :return: cost值和updates字典。字典包含权重和偏置的更新，同时也包含储存固定链的
                共享变量的更新。
        """
        #计算正项
        pre_sigmoid_ph,ph_mean,ph_sample=self.sample_h_given_v(self.input)
        #决定初始化固定链的方法:对于CD，采用全新生成隐含样本；对于PCD，从链的旧状态获得
        if persistent is None:
            chain_start=ph_sample
        else:
            chain_start=persistent

        #计算负项
        #为了执行CD-k/PCD-k，我们需要循环执行一步Gibbs采样k次
        #阅读Theano tutorial中scan的介绍获得更多内容
        # http://deeplearning.net/software/theano/library/scan.html
        #scan返回完整的Gibbs链
        [pre_sigmoid_nvs,nv_means,nv_samples,
         pre_sigmoid_nhs,nh_means,nh_samples],updates=\
            theano.scan(self.gibbs_hvh,
                    #下面字典中前5项为None,表示chain_start与初始状态中第六个输出量有关
                    outputs_info=[None,None,None,None,None,chain_start],
                    n_steps=k)
        #计算RBM参数的梯度,只需要从链末端采样
        chain_end=nv_samples[-1]

        cost=T.mean(self.free_energy(self.input))-T.mean(self.free_energy(chain_end))
        #因为chai_end是符号变量，而我们只根据链最末端的数据求梯度，所有指定chain_end为常数
        gparams=T.grad(cost,self.params,consider_constant=[chain_end])

        #构造更新字典
        for gparam,param in zip(gparams,self.params):
            #确保学习率lr的数据类型正确
            updates[param]=param-gparam*T.cast(lr,dtype=theano.config.floatX)

        #RBM是深度网络的一个模块时,更新perisistent
        if persistent:
            #只有persistent为共享变量时才运行
            updates[persistent]=nh_samples[-1]
            #伪似然函数是PCD的一个较好的代价函数
            monitoring_cost=self.get_pseudo_likehood_cost(updates)
        #RBM是标准网络
        else:
            #重构交叉熵是CD的一个较好的代价函数
            monitoring_cost=self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])

        return monitoring_cost,updates

    #伪似然函数的随机近似算法
    def get_pesudo_likehood_cost(self,updates):
        #定义表达式p{x_i|x{\i}}的索引i
        bit_i_idx=theano.shared(value=0,name='bit_i_idx')
        #二值化输入图像，将它近似到最近的整数
        xi=T.round(self.input)  #?????????input已经是二值化了，为什么还要round？
        #给定bit设置后计算自由能
        fe_xi=self.free_energy(xi)
        #在矩阵xi中反转变量x_i，并保存其他变量x_{\i}。
        # 等价于运算：xi[:,bit_i_idx]=1-xi[:,bit_i_idx]
        #相当于做运算：xi[:,bit_i_idx]
        #设定所有值在xi_flip运算，而不是xi上运算
        xi_flip=T.set_subtensor(xi[:,bit_i_idx],1-xi[:,bit_i_idx])
        #等价运算：e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost=T.mean(self.n_visible*T.log(T.nnet.sigmoid(xi_flip-fe_xi)))
        #将bit_i_idx%number增加到updates中，随机选取下次的索引
        updates[bit_i_idx]=(bit_i_idx+1)%self.n_visible
        return cost

    def get_reconstruction_cost(self,updates,pre_sigmoid_nv):
        """近似重构误差算法
        注意函数要求未sigmoid激活的值作为输入量。如果想深入了解这样做的原因，那么
        需要了解Theano的工作原理。当编译Theano函数时，计算图中输入量的速度和稳定性
        得到优化，这是通过改变子图中若干部分实现的。这样的优化代表softplus中log(sigmoid(x))项。对于交叉熵，当sigmoid值大于30(结果趋于1)，
        就需要这样的优化。当sigmoid值小于-30(结果趋于0)，则Theano计算log(0)，最终代价为-inf
        或者NaN。通常情况下，softplus中log(sigmoid(x))项会得到正常值。但这里遇到特殊情况：
        sigmoid在scan优化内部，log在外部。因此，Theano会执行log(scan(…))而不是log(sigmoid(…))，
        也不会进行优化。我们找不到替代scan中sigmoid的方法，因为只需要在最后一步执行。最简单有效
        的办法是输出未sigmoid的值，在scan之外同时应用log和sigmoid。

        :param updates:
        :param pre_sigmoid_nv:
        :return:
        """

#测试函数
if __name__=='__main__':
    rbm=RBM()
