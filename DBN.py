#-*-coding: utf-8 -*-
__author__ = 'Administrator'

import cPickle
import gzip
import sys
import time
import numpy
import os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data,LogisticRegression
from mlp import HiddenLayer
from rbm import RBM

class DBN(object):
    """
    深度置信网络
    深度置信网络是将若干RBMs堆叠组成的。第i层RBM的隐层是第i+1层的输入。
    第一层RBM的输入是网络的输入，最后一层RBM的隐层是网络的输出。当用于分类时，
    DBN顶部添加一个logistic回归，变成了MLP。
    """
    def __init__(self,numpy_rng,theano_rng=None,n_ins=784,
                 hidden_layers_sizes=[500,500],n_outs=10):
        """
        该类可实现可变层数的DBN

        :param numpy_rng: numpy.random.RandomState  用于初始化权重的numpy随机数
        :param theano_rng: theano.tensor.shared_randomstreams.RandomStreams
                            如果输入为None
        :param n_ins: int DBN输入量的维度
        :param hidden_layers_size: list 隐层输入量的维度
        :param n_outs: int 网络输出量的维度
        :return:
        """
        self.sigmoid_layers=[]
        self.rbm_layers=[]
        self.params=[]
        self.n_layers=len(hidden_layers_sizes)
        assert self.n_layers>0

        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2**30))
        #设置符号变量
        self.x=T.matrix('x')
        self.y=T.ivector('y')

        #DBN是一个MLP，中间层的权重是在不同的RBM之间共享的。
        #首先构造DBN为一个深层多感知器。在构造每个sigmoid层时，
        #同样构造RBM与之共享变量。在预训练阶段，需要训练三个RBM(同样改变MLP的权重，
        #微调阶段，通过在MLP上随机梯度下降法完成DBN训练。

        for i in xrange(self.n_layers):
            #构造sigmoid层，
            #对于第一层，输入量大小是网络的输入量大小
            #对于其它层，输入量大小是下层隐层单元的数量
            if i==0:
                input_size=n_ins
            else:
                input_size=hidden_layers_sizes[i-1]
            #对于第一层，输入是网络的输入
            #对于其它层，输入是下层隐层的激活函数值
            if i==0:
                layer_input=self.x
            else:
                layer_input=self.sigmoid_layers[i-1].output
            #定义sigmoid函数
            sigmoid_layer=HiddenLayer(rng=numpy_rng,input=layer_input,n_in=input_size,
                                      n_out=hidden_layers_sizes[i],activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            #sigmoid_layers的参数是DBN的参数。而RBM中可见层的偏置只是RBM的参数，而不属于DBN
            self.params.extend(sigmoid_layer.params)

            #构造RBM共享权重
            rbm_layer=RBM(input=layer_input,n_visible=input_size,n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,hbias=sigmoid_layer.b,numpy_rng=numpy_rng,theano_rng=theano_rng)

            self.rbm_layers.append(rbm_layer)

        #添加logistic到网络的顶部
        self.logLayer=LogisticRegression(input=self.sigmoid_layers[-1].output,
                                         n_in=hidden_layers_sizes[-1],n_out=n_outs)
        self.params.extend(self.logLayer.params)

        #计算微调阶段的代价函数，定义为logistic回归(输出)层的负对数似然函数
        self.finetune_cost=self.logLayer.negative_log_likelihood(self.y)

        #给定self.x和self.y，计算每个minibatch的误差
        self.errors=self.logLayer.errors(self.y)

    def pretraining_functions(self,train_set_x,batch_size,k):
        """
        生成函数列表，在给定层计算一步梯度下降。函数要求输入minibatch索引，
        重复训练RBM,并在所有的minibatch调用相关函数。

        train_set_x: theano.tensor.TensorType 训练集的特征
        batch_size: int minibatch的大小
        k:  int CD-k/PCD-k中Gibbs采样步数
        """
        index=T.lscalar('index') #minibatch的索引
        learning_rate=T.scalar('lr') #学习率
        #bathes数量
        n_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
        #给定index后，起始batch
        batch_begin=index*batch_size
        #给定index后，终止batch
        batch_end=batch_begin+batch_size

        pretrain_fns=[]
        for rbm in self.rbm_layers:  #依次训练每个RBM
            #获得代价值和更新列表
            #使用CD-k(这里persisitent=None)，训练每个RBM
            cost,updates=rbm.get_cost_updates(learning_rate,persistent=None,k=k)

            #定义thenao函数,需要将learning_rate转换为tensor类型
            fn=theano.function(inputs=[index,theano.Param(learning_rate,default=0.1)],
                               outputs=cost,updates=updates,
                               givens={self.x:train_set_x[batch_begin:batch_end]})
            #将'fn'增加到list列表中
            pretrain_fns.append(fn)
        return pretrain_fns


    def build_finetune_function(self,datasets,batch_size,learning_rate):
        """
        构造训练函数执行一步微调，构造验证函数计算验证集合一个batch的误差
        构造测试函数计算测试集合一个batch的误差
        datasets: theano.tensor.TensoType 数据集合
        batch_size: int minibatch大小
        laerning_rate: float 微调阶段的学习率

        """
        (train_set_x,train_set_y)=datasets[0]
        (valid_set_x,valid_set_y)=datasets[1]
        (test_set_x,test_set_y)=datasets[2]

        #计算训练、验证、测试的minibatch数量
        n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size
        n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size
        #minibath索引的符号变量
        index=T.lscalar('index')
        #计算梯度下降率
        gparams=T.grad(self.finetune_cost,self.params)
        #生成更新列表
        updates=[]
        for param,gparam in zip(self.params,gparams):
            updates.append((param,param-gparam*learning_rate))


        #定义训练函数
        train_fn=theano.function(inputs=[index],outputs=self.finetune_cost,updates=updates,
                                 givens={self.x:train_set_x[index*batch_size:(index+1)*batch_size],
                                         self.y:train_set_y[index*batch_size:(index+1)*batch_size]})
        #定义一个minibatch上的验证函数
        valid_score_i=theano.function(inputs=[index],outputs=self.errors,updates=updates,
                                     givens={self.x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                             self.y:valid_set_y[index*batch_size:(index+1)*batch_size]})
        #定义一个minibatc上的测试函数
        test_score_i=theano.function(inputs=[index],outputs=self.errors,updates=updates,
                                     givens={self.x:test_set_x[index*batch_size:(index+1)*batch_size],
                                             self.y:test_set_y[index*batch_size:(index+1)*batch_size]})

        #定义整个验证集合上的验证函数
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        #定义整个测试集合上的测试函数
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn,valid_score,test_score



#---------------测试函数----------
def test_DBN(finetune_lr=0.1,pretraining_epochs=100,
             pretrain_lr=0.01,k=1,training_epoch=1000,
             dataset='./data/mnist.pkl.gz',batch_size=10):
    """
    定义训练和测试深度置信网络的函数
    :param finetune_lr: float   微调阶段的学习率
    :param pretraining_epochs: int 进行预训练的迭代次数
    :param pretrain_lr: float 预训练阶段的学习率
    :param training_epoch: int 进行训练的迭代次数
    :param dataset: str 数据集的路径
    :param batch_size: int minibatch的大小
    :return:
    """
    #########################
    #     模型初始化过程     #
    #########################
    datasets=load_data(dataset)
    train_set_x,train_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x,test_set_y=datasets[2]

    #计算minibatch的数量
    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
    #numpy生成的随机数种子
    numpy_rng=numpy.random.RandomState(123)
    print '...building the model'
    #实例化DBN，有三个隐层
    dbn=DBN(numpy_rng,n_ins=28*28,hidden_layers_sizes=[1000,1000,1000],n_outs=10)

    #########################
    #     模型预训练过程     #
    #########################
    print "...getting the pretraining functions"
    pretraining_fns=dbn.pretraining_functions(train_set_x=train_set_x,batch_size=batch_size,k=k)

    print "...pretraining the model"
    start_time=time.clock()
    #逐层预训练
    for i in xrange(dbn.n_layers):
        #遍历训练次数
        for epoch in xrange(pretraining_epochs):
            #遍历每个minibatch
            c=[] #定义储存RBM中cost的列表
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))
            print "pretraining layer %i, epoch %i,cost " %(i,epoch),
            print numpy.mean(c)
    end_time=time.clock()
    print >>sys.stderr,("The pretraining code for file "+
                        os.path.split(__file__)[1]+
                        " ran for %0.2fm")%((end_time-start_time)/60.)

    #########################
    #       模型微调过程     #
    #########################
    #构造微调过程的训练函数、验证函数和测试函数
    print "...getting finetuning functions"
    train_fn,valid_model,test_model=dbn.build_finetune_function(datasets=datasets,
                                    batch_size=batch_size,learning_rate=finetune_lr)

    print "...finetuning the model"
    #提前结束的参数设置
    patience=4*n_train_batches
    patience_increase=2.
    improvement_threshold=0.995 #每次优化效果阈值
    #在验证集合检查minibatch，
    #该程序中每个epoch都要检查
    validation_frequency=min(n_train_batches,patience/2)
    #微调过程初始参数设置
    best_params=None
    best_validation_loss=numpy.inf
    test_score=0.
    start_time=time.clock()
    done_looping=False
    epoch=0

    #设置终止条件：大于设定的迭代次数或者达到don_looping
    while(epoch<training_epoch)and(not done_looping):
        epoch+=1
        for minibath_index in xrange(n_train_batches):
            minibath_avg_lost=train_fn(minibath_index)
            iter=(epoch-1)*n_train_batches+minibath_index #当前minibath的总索引

            #判断是否达到validation_frequency
            if (iter+1)%validation_frequency==0:
                validation_losses=valid_model()
                this_validation_loss=numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%') %\
                     (epoch,minibath_index+1,n_train_batches,this_validation_loss*100)

                #如果当前代价值优于历史代价值
                if this_validation_loss<best_validation_loss:
                    if this_validation_loss<best_validation_loss*improvement_threshold:
                        patience=max(patience,iter*patience_increase)

                    #保存最优验证值和minibatch索引
                    best_validation_loss=this_validation_loss
                    best_iter=iter

                    #在测试集进行测试
                    test_losses=test_model()
                    test_score=numpy.mean(test_losses)
                    print "epoch %i, minibath %i/%i, test error of best model %f %%" %\
                    (epoch,minibath_index+1,n_train_batches,test_score*100)

            if patience<=iter:
                done_looping=True
                break
    end_time=time.clock()
    print "Optimizaiton complete with best validation score of %f %%,"\
          "with best performance %f %%"%(best_validation_loss*100.,test_score*100.)

    print >> sys.stderr, ("The fine tuning code for file "+
                          os.path.split(__file__)[1]+
                          " ran for %.2fm")%((end_time-start_time)/60.)











if __name__=='__main__':
    test_DBN()