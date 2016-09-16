#-*- coding:utf-8 -*-

import numpy
from scipy import linalg
import theano

from hmc import HMC_sampler

def sampler_on_nd_gaussian(sampler_cls,burnin,n_samples,dim=10):
    batchsize=3
    rng=numpy.random.RandomState(123)

    #定义高斯的期望和协方差
    mu=numpy.array(rng.rand(dim)*10,dtype=theano.config.floatX)
    cov=numpy.array(rng.rand(dim,dim),dtype=theano.config.floatX)
    cov=(cov+cov.T)/2
    cov[numpy.arange(dim),numpy.arange(dim)]=1.0
    cov_inv=linalg.inv(cov)  #cov的逆矩阵

    #定义多变量高斯函数的能量函数
    def gaussian_energy(x):
        return 0.5*(theano.tensor.dot((x-mu),cov_inv)*(x-mu)).sum(axis=1)

    #定义位置的共享变量
    positon=rng.randn(batchsize,dim).astype(theano.config.floatX)
    positon=theano.shared(positon)

    #创建HMC样本
    sampler=sampler_cls(positon,gaussian_energy,
                        initial_stepsize=1e-3,stepsize_max=0.5)

    #开始HMC仿真
    garbage=[sampler.draw() for r in xrange(burnin)]
    #'n_samples':返回3D张量 [n_samples,batchsize,dim]
    _samples=numpy.asarray([sampler.draw() for r in xrange(n_samples)])
    #将样本展开，[n_samples*batchsize,dim]
    samples=_samples.T.reshape(dim,-1).T

    print '************ TARGET VALUES **************'
    print 'target mean: ',mu
    print  'target cov:\n ',cov

    print '****** EMPIRICAL MEAN/COV USING HMC ******'
    print 'empirical mean: ', samples.mean(axis=0)
    print 'empirical_cov:\n', numpy.cov(samples.T)

    print '****** HMC INTERNALS ******'
    print 'final stepsize', sampler.stepsize.get_value()
    print 'final acceptance_rate', sampler.avg_acceptance_rate.get_value()

    return sampler


def test_hmc():
    sampler = sampler_on_nd_gaussian(HMC_sampler.new_from_shared_positions,
            burnin=1000, n_samples=1000, dim=5)
    assert abs(sampler.avg_acceptance_rate.get_value() -
               sampler.target_acceptance_rate) < .1
    assert sampler.stepsize.get_value() >= sampler.stepsize_min
    assert sampler.stepsize.get_value() <= sampler.stepsize_max





