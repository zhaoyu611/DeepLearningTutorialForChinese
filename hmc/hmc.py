#-*- coding:utf-8 -*-

import numpy
from theano import function ,shared
from thenao import tensor as TT
import theano

sharedX=lambda X, name : \
        shared(numpy.asarray(X,dtype=theano.config.floatX),name=name)

#定义动能函数
#input:
#       vel: matrix的符号变量，每行为速度矢量
#output:
#       vector的符号矢量 第i个元素是第i个速度的动能
def kinetic_energy(vel):
    return 0.5*(vel**2).sum(axis=1)

#给定位置和速度，返回hamilton函数(动能与势能之和)
#input:
#       pos:matrix的符号变量，每行为位置矢量
#       vel:matrix的符号变量，每行为速度矢量
#       energy_fn:python函数 给定位置信息后，输出势能
#output：
#       theano vector 其中第i个元素表示第i个位置和第i个速度的hanmilton函数
def hamiltonian(pos,vel,energy_fn):
    return kinetic_energy(vel)+energy_fn(pos)

#进行metropolis_hastings接受拒绝计算
#input:
#      energy_prev: theano vector t时刻的能量函数
#      engergy_next: theano vector t+1时刻的能量函数
#      s_rng: RandomStreams 用于生成随机数的stream对象
#output:
#       返回为真，则是接受，
def metropolis_hastings_accept(energy_prev,energy_next,s_rng):
    ediff=energy_next-energy_prev
    return (TT.exp(ediff)-s_rng.uniform(size=energy_prev.shape))>=0

#执行'n_steps'步使用hamilton动态函数更新后，返回最终的(位置。速度)信息。
#input:
#       initial_pos: theano matrix  初始位置
#       initial_vel: thenao matrix  初始速度
#       stepsize: theano scalar 步长
#       n_steps: theano scalar 总步数
#       energy_fn: python function 计算势能的函数
#output:
#       rval1: thenao matrix 最终位置
#       rval2: theano matrix 最终速度
def simulate_dynamics(initial_pos,initial_vel,stepsize,n_steps,energy_fn):
    #一步hamilton动态函数更新
    #input:
    #       pos: theano matrix t时刻的位置信息
    #       vel: theano matrix t时刻的速度信息
    #       step: thenao scalar 步长
    #output:
    #       rval1: [matrix,matrix] 新的位置pos(t+stepsize) 新的速度vel(t+stepsize)
    #       rval2: dictionary 用于Scan操作的updates
    def leapfrog(pos,vel,step):
        #根据pos(t)和vel(t-stepsize/2)计算vel(t+stepsize/2)
        dE_pos=TT.grad(energy_fn(pos).sum(),pos)
        new_vel=vel-step*dE_pos
        new_pos=pos+step*new_vel
        return [new_pos,new_vel],{}

    #计算t+stepsize/2时刻的速度
    initial_energy=energy_fn(initial_pos)
    dE_dpos=TT.grad(initial_energy.sum(),initial_pos)
    vel_half_step=initial_vel-0.5*stepsize*dE_dpos
    #计算t+stepsize时刻的位置
    pos_full_step=initial_pos+stepsize*vel_half_step

    #进行leapfrog更新：使用scan操作
    #vel(t+(m-1/2)*stepsize)和pos(t+m*stepsize) 其中m为[2,n_step]
    (all_pos,all_vel),scan_updates=theano.scan(leapfrog,
                                outputs_info=[dict(initial=pos_full_step),
                                              dict(initial=vel_half_step),
                                              ],
                                non_sequences=[stepsize],
                                n_steps=n_steps-1)
    final_pos=all_pos[-1]
    final_vel=all_vel[-1]
    #Scan函数返回更新字典，Scan函数从RandomStream中采样
    #在编译Theano函数时，使用更新字典，从而避免每次调用函数时生成随机数。
    #然后在本程序中，有意识的忽略“scan_updates”，因为我们知道它是空的
    assert not scan_updates

    #scan返回的最后的速度值 vel((t +# (n_steps - 1 / 2) * stepsize))
    #此时，再多进行一次操作，返回vel(t + n_steps * stepsize)
    energy=energy_fn(final_pos)
    final_vel=final_vel-0.5*stepsize*TT.grad(energy.sum(),final_pos)

    #返回最终的位置和速度
    return final_pos,final_vel


#本函数执行一步HMC采样。首先，由单变量高斯分布对速度采样，然后使用Hamilton动态函数
# 执行'n_steps'蛙跳更新，并使用Metropolis-Hastings执行接受拒绝检验。
#input：
#       s_rng: thenao.shared.random.stream 生成随机速度
#       positions: theano.shared.matrix 位置矩阵，每行为位置矢量
#       energy_fn: python函数 计算给定位置的势能
#       stepsize: theano.shared.scalar  'n_steps'步HMC仿真的步长
#       n_steps: int 仿真步数
#output:
#       rval1:bool 结果为真，则接受move，否则拒绝move
#       rval2: thenao.matrix  每行为新的位置
def hmc_move(s_rng,positions,energy_fn,stepsize,n_steps):
    #速度的随机采样
    initial_vel=s_rng.normal(size=positions.shape)
    #执行Hamilton动态函数仿真
    final_pos,final_vel=simulate_dynamics(initial_pos=positions,
                                          initial_vel=initial_vel,
                                          stepsize=stepsize,
                                          n_steps=n_steps,
                                          energy_fn=energy_fn)
    #基于联合分布计算接受/拒绝
    accept=metropolis_hastings_accept(energy_prev=hamiltonian(positions,initial_vel,energy_fn),
                                      energy_next=hamiltonian(positions,initial_vel,energy_fn),
                                      s_rng=s_rng)

    return accept,final_pos

#函数执行'n_step'步HMC采样(hmc_move函数)。本函数生成了用于'simulate'函数的更新字典，
#更新包括：位置(如果move被接受)，stepsize（计算给定目标的接受率），平均接受率(计算moving平均)
#input:
#       positions: theano.matrix 每行为旧位置
#       stepsize: thenao.scalar 当前步长
#       avg_acceptance_rate: theano.scalar 当前平均接受率
#       final_pos: theano.matrix  每行为新位置
#       accept: theano.scalar bool型，HMCmove是否被接受
#       target_acceptance_rate: float  目标接受率
#       stepsize_inc: float 步长增长率
#       stepsize_dec: float 步长下降率
#       stepsize_min: float 最小步长
#       stepsize_max: float 最大步长
#       avg_acceptance_slowness: float 指数平均moving的平均接受率
#       (1-avg_acceptance_slowness)是最新观测值的权重
def hmc_updates(positions,stepsize,avg_acceptance_rate,final_pos,accept,
                target_acceptance_rate,stepsize_inc,stepsize_dec,
                stepsize_min,stepsize_max,avg_acceptance_slowness):

        ###################
        ######位置更新######
        ###################
        #将'accept'由scalar扩展为tensor，与final_pos有相同维度,dimshuffle不太懂
        accept_matrix=accept.dimshuffle(0,*(('x',) * (final_pos.ndim - 1)))
        #如果accept为True，则更新'final_pos'，否则保持不变
        new_positions=TT.switch(accept_matrix,final_pos,positions)

        ###################
        ###步长更新#########
        ###################
        #如果接受率太低，或样本噪点太多，那么减小步长；如果接受率太高，或样本过于保守，
        #那么增大步长。
        _new_stepsize=TT.switch(avg_acceptance_rate>target_acceptance_rate,
                                stepsize*stepsize_inc,
                                stepsize*stepsize_dec)
        #保持stepsize在[stepsize_min,stepsize_max]区间内
        new_stepsize=TT.clip(_new_stepsize,stepsize_min,stepsize_max)
        ###################
        ##接受率更新########
        ###################
        #执行指数平均moving
        mean_dtype=theano.scalar.upcast(accept.dtype,avg_acceptance_rate.dtype)
        new_acceptance_rate=TT.add(
            avg_acceptance_slowness*avg_acceptance_rate,
            (1-avg_acceptance_slowness)*accept.mean(dtype=mean_dtype))

        return [(positions,new_positions),
                (stepsize,new_stepsize),
                (avg_acceptance_rate,new_acceptance_rate)]

#混合蒙特卡洛采样的封装函数。该函数建立符号图执行HMC仿真（使用'hmc_move'和'hmc_updates'）。
#该图在'simulate'函数中编译，'simulate'函数执行仿真，其中更新值要求共享变量。
#用户通过'draw'函数中先进蒙特卡洛法采样，函数中依次调用‘simulate’和 ‘get_position’函数返回当前采样值
#超参数与Marc'Aurelio'编写的'train_mcRBM.py'文件中超参数相同(代码见Marc'Aurelio'的个人主页)。
class HMC_sampler(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    #input:
    #       shared_positions:theano.tensor.matrix 保存很多粒子初始位置的一维矩阵
    #       energy_fn: 调用energy_fn(positions)函数，返回能量矢量，长度为batchsize
    #                  能量矢量之和必须可微(使用theano.tensor.grad)，用于HMC采样
    @classmethod
    def new_from_shared_positions(cls,shared_positions,energy_fn,
                                  initial_stepsize=0.01,target_acceptance_rate=0.9,n_steps=20,
                                  stepsize_dec=0.88,
                                  stepsize_min=0.001,
                                  stepsize_max=0.25,
                                  stepsize_inc=1.02,
                                  avg_acceptance_slowness=0.9, #用于生成avg，如果值为1.0,则不移动
                                  seed=12345):
        batchsize=shared_positions.shape[0]
        #定义符号变量
        stepsize=sharedX(initial_stepsize,'hmc_stepsize')
        avg_acceptance_rate=sharedX(target_acceptance_rate,'avg_acceptance_rate')
        s_rng=TT.shared_randomstreams.RandomStreams(seed)

        #定义'n_step'步HMC仿真
        accept,final_pos=hmc_move(
            s_rng,
            shared_positions,
            energy_fn,
            stepsize,
            n_steps)

        #定义更新字典，用于每次'simulate'函数
        simulate_updates=hmc_updates(
            shared_positions,
            stepsize,
            avg_acceptance_rate,
            final_pos=final_pos,
            accept=accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_inc=stepsize_inc,
            stepsize_dec=stepsize_dec,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_rate=avg_acceptance_rate)

        #编译函数
        simulate=function([],[],updates=simulate_updates)

        #创建包含以下属性的HMC_sampler对象：
        return cls(
                    positions=shared_positions,
                    stepsize=stepsize,
                    stepsize_min=stepsize_min,
                    stepsize_max=stepsize_max,
                    avg_acceptance_rate=avg_acceptance_rate,
                    target_acceptance_rate=target_acceptance_rate,
                    s_rng=s_rng,
                    _updates=simulate_updates,
                    simulte=simulate)

    #执行'n_steps'HMC仿真后，返回新的位置
    #input:
    #       kwargs: dictionary 使用'get_values()'函数传递共享变量(self.positions)
    #       例如，如果不想复制共享变量，则设置'borrow=False'
    #output:
    #       rval: numpy.matrix 矩阵维度与'iniital_position'相同
    def draw(self,**kwargs):
        self.simulate()
        return self.positions.get_value(borrow=False)













