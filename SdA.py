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
                 hidden_layers_sizes=[500,500],n_outs=10,
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
        self.y=T.ivector('y') #由[int]型标签组成的一维向量

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
                        input=layer_input,
                        n_visible=input_size,
                        n_hidden=hidden_layers_sizes[i],
                        W=sigmoid_layer.W,
                        bhid=sigmoid_layer.b
                        )
            self.dA_layers.append(dA_layer)

        #在MLP顶部加上losgistic层
        self.logLayer=LogisticRegression(
                        input=self.sigmoid_layers[-1].output,
                        n_in=hidden_layers_sizes[-1],n_out=n_outs)
        self.params.extend(self.logLayer.params)
        #建立函数，执行一步微调
        #定义第二步训练的代价：负log函数
        self.finetune_cost=self.logLayer.negative_log_likelihood(self.y)
        #分别对模型中参数计算梯度
        #给定self.x和self.y，定义每个minibatch上的误差的符号变量
        self.errors=self.logLayer.errors(self.y)

    def pretraining_function(self,train_set_x,batch_size):
        '''
        生成函数列表，每个函数执行一层中dA的训练,返回预训练的函数列表
        函数输入是minibatch的索引，在所有的minibatch执行相同的训练
        train_set_x: theano.tensor.TensorType   训练dA的数据点(共享变量)
        batch_size: int  [mini]batch大小

        '''
        #[mini]batch的索引
        index=T.lscalar('index')
        corruption_level=T.scalar('corruption') #corruption百分比
        learning_rate=T.scalar('lr') #学习率
        #batch数量
        n_bathes=train_set_x.get_value(borrow=True).shape[0]/batch_size
        #给定index后，起始的
        # batch
        batch_begin=index*batch_size
        #给定index后，结束的batch
        batch_end=batch_begin+batch_size

        pretrain_fns=[]
        for dA in self.dA_layers: #遍历dA
            #创建代价列表和更新列表
            cost,updates=dA.get_cost_updates(corruption_level,
                                            learning_rate)
            #创建theano函数
            fn=theano.function(inputs=[index,
                            theano.Param(corruption_level,default=0.2),
                            theano.Param(learning_rate,default=0.1)],
                                outputs=cost,
                                updates=updates,
                                givens={self.x:train_set_x[batch_begin:
                                                           batch_end]})
            #将fn添加到函数列表
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self,datasets,batch_size,learning_rate):
        '''
        创建"train"函数执行一步微调；"validate"函数计算验证集合中batch的误差；
        "test"函数计算测试集合中batch误差

        :param datasets: list of pairs of theano.tensor.TensorType
                         #包含所有datasets的列表，每3个元素为一个组：
                         依次为'train'、'valid'、'test'。每个元素又
                         包含两个theano变量：数据特征和标签
        :param batch_size: int  minibatch的大小
        :param learning_rate:float  微调阶段的learning_rate
        :return:
        '''

        (train_set_x,train_set_y)=datasets[0]
        (valid_set_x,valid_set_y)=datasets[1]
        (test_set_x,test_set_y)=datasets[2]

        #分别计算training、validation、testing的minibatch的数量
        n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches/=batch_size
        n_test_batches=test_set_x.get_value(borrow=True).shape[0]
        n_test_batches/=batch_size

        index=T.lscalar('index') #[mini]batch的索引
        #分别计算模型参数的梯度
        gparams=T.grad(self.finetune_cost,self.params)
        #计算微调更新参数列表
        updates=[]
        for param,gparam in zip(self.params,gparams):
            updates.append((param,param-gparam*learning_rate))

        train_fn=theano.function(inputs=[index],
                                 outputs=self.finetune_cost,
                                 updates=updates,
                                 givens={
                                     self.x:train_set_x[index*batch_size:
                                                        (index+1)*batch_size],
                                     self.y:train_set_y[index*batch_size:
                                                        (index+1)*batch_size]},
                                 name='train')
        test_score_i=theano.function([index],self.errors,
                                     givens={
                                         self.x:train_set_x[index*batch_size:
                                                         (index+1)*batch_size],
                                         self.y:train_set_y[index*batch_size:
                                                        (index+1)*batch_size]},
                                     name='test')
        valid_score_i=theano.function([index],self.errors,
                                      givens={
                                          self.x:train_set_x[index*batch_size:
                                                            (index+1)*batch_size],
                                          self.y:train_set_y[index*batch_size:
                                                            (index+1)*batch_size]},
                                      name='valid')
        #创建函数遍历整个验证集合
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches) ]
        #创建函数遍历整个测试集合
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
        return train_fn,valid_score(),test_score()

def test_SdA(finetune_lr=0.1,pretraining_epochs=15,
             pretrain_lr=0.001,training_epochs=1000,
             dataset='./data/mnist.pkl.gz',batch_size=1):
    '''
    创建函数，训练和测试随机降噪自编码器，实验数据为MNINST


    :param finetune_lr: float   微调阶段的学习率(随机梯度下降的影响因素)
    :param pretraining_epochs:int   预训练的迭代次数
    :param pretrain_lr: float   预训练阶段的学习率
    :param training_epochs: int 整个训练的最大次数
    :param dataset: string 数据集的路径
    :param batch_size: batch大小
    :return:
    '''
    datasets=load_data(dataset)

    train_set_x,training_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x,test_set_y=datasets[2]
    #计算训练的minibatch个数
    n_train_batches=train_set_x.get_value(borrow=True).shape[0]
    n_train_batches/=batch_size

    #生成随机数
    numpy_rng=numpy.random.RandomState(89677)
    print '...building the model'
    #实例化栈式自编码类
    sda=SdA(numpy_rng=numpy_rng,n_ins=28*28,
            hidden_layers_sizes=[1000,1000,1000],n_outs=10)

    #########################
    #        预训练过程      #
    #########################
    print '...getting the pretraining functions'
    pretraining_fns=sda.pretraining_function(train_set_x=train_set_x,
                                                 batch_size=batch_size)
    print '... pre-training the model'
    stat_time=time.clock()
    #逐层训练
    corruption_levels=[0.1,0.2,0.3]
    #对每层进行预训练
    for i in xrange(sda.n_layers):
        #进行n次预训练
        for epoch in xrange(pretraining_epochs):
            #遍历整个训练集合
            c=[]
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch)
            print numpy.mean(c)
    end_time=time.clock()
    print >>sys.stderr,('The pretraining code for file'+
                        os.path.split(__file__)[1]+
                        'ran for %0.2fm')%((end_time-stat_time)/60)
    ########################
    #       微调模型        #
    ########################
    #创建模型的训练函数、验证函数和测试函数
    print '...getting the finetuning functions'
    train_fn,validate_model,test_model=sda.build_finetune_functions(
                                        datasets=datasets,batch_size=batch_size,
                                        learning_rate=finetune_lr)
    print '... finetuning the model'
    #提前终止的条件参数
    patience=10*n_train_batches # 通常训练中忽略该条件
    patience_increase=2.0 #当发现新的最优值时，patience增加值为2.0
    improvement_threshold=0.995 #提高的阈值限定
    validation_frequency=min(n_train_batches,patience/2) #在验证集合的上检查minibatch的次数
                                                         #在本实验中，minibatch为1，
                                                         # 每次迭代都检查
    best_params=None
    best_validation_loss=numpy.inf
    test_score=0.
    start_time=time.clock()

    done_looping=False
    epoch=0

    while (epoch<training_epochs) and (not done_looping):
        epoch=epoch+1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost=train_fn(minibatch_index)
            iter=(epoch-1)*n_train_batches+minibatch_index  #数据的编号
            #每个minibatch都输出验证集代价值,该实验中minibatch为1
            if (iter+1)%validation_frequency==0:
                validation_losses=validate_model()
                this_validation_loss=numpy.mean(validation_losses)
                print ('epoch %i, minibatch %i/%i, validation error %f %%' %
                       (epoch,minibatch_index+1,n_train_batches,
                        this_validation_loss*100))

                #如果我们得到当前最优验证得分，则更新
                if this_validation_loss<best_validation_loss:
                    #增加patience，如果loss improvement足够好
                    if (this_validation_loss<best_validation_loss*
                        improvement_threshold):
                        patience=max(patience,iter*patience_increase)

                    #保存最优验证值和编号
                    best_validation_loss=this_validation_loss
                    best_iter=iter

                    #测试集合上进行测试
                    test_losses=test_model()
                    test_score=numpy.mean(test_losses)
                    print ('epoch %i, minibath %i/%i, test error of '
                           'best model %f %%') %(epoch,minibatch_index+1,
                            n_train_batches,test_score*100.)

            if patience<=iter: #设置终止条件
                done_looping=True
                break
    end_time=time.clock()
    print ('optimization complete with best validation score of %f %%,'
            'with test performance %f %%') %(best_validation_loss%100,test_score*100)
    print >>sys.stderr, ('The training code for file'+
                         os.path.split(__file__)[1]+
                         'ran for %0.2fm')%((end_time-start_time)/60.)


if __name__=='__main__':
    test_SdA()


