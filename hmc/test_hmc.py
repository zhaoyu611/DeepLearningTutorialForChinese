#-*- coding:utf-8 -*-

import numpy
from theano import function ,shared
from thenao import tensor as TT
import theano

sharedX=lambda :X ,name:\
        shared(numpy.asarray(X,dtype=theano.config.floatX),name=name)
