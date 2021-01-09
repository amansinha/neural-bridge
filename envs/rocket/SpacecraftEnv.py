import tensorflow as tf
import sys
import numpy as np
import h5py
class SpacecraftEnv(object):
 MAXTIME=100
 SCALE=1.0e4
 EWIND1=np.array([1.0,1.0,1.0])/np.sqrt(3.0),
 EWIND2=np.array([0,1.0,0]),
 EWIND3=np.array([1.0,0,0]),
 def __init__(self,dt=0.2,g=10,m=5.0e4,Fmax=1.0e6,p0=np.array([500.0,500.0,1000.0]),v0=np.array([-100.0,0,-100.0]),alpha=0.5,Kp=5.0e4,Kv=3.0e5,Cwind=200.0,optimal_file_path=None):
  self.dt=dt
  self.g=g
  self.m=m/SpacecraftEnv.SCALE
  self.Fmax=Fmax/SpacecraftEnv.SCALE
  self.p0=p0.astype(np.float32)
  self.v0=v0.astype(np.float32)
  self.alpha=alpha
  self.Kp=Kp/SpacecraftEnv.SCALE/8
  self.Kv=Kv/SpacecraftEnv.SCALE/8
  self.Cwind =Cwind/SpacecraftEnv.SCALE
  if optimal_file_path is not None:
   f=h5py.File(optimal_file_path,'r')
   fstar=f['fstar'][()]
   pstar=f['pstar'][()]
   vstar=f['vstar'][()]
   leny=SpacecraftEnv.MAXTIME-fstar.shape[0]
   assert(leny>=0)
   if leny>0:
    zeromat=np.zeros(shape=(leny,fstar.shape[1]))
    fstar=np.concatenate([fstar,zeromat],axis=0)
    pstar=np.concatenate([pstar,zeromat],axis=0)
    vstar=np.concatenate([vstar,zeromat],axis=0)
   self.fstar=fstar.astype(np.float32)/SpacecraftEnv.SCALE
   self.pstar=pstar.astype(np.float32)
   self.vstar=vstar.astype(np.float32)
 def step_multiple(self,position,velocity,index,noise):
  height=tf.expand_dims(self.pstar[index,2],axis=1)
  wind=self.Cwind*height*noise
  ep=position-self.pstar[index,:]
  ev=velocity-self.vstar[index,:]
  f=self.fstar[index,:]-self.Kp*ep-self.Kv*ev+wind
  f=tf.clip_by_norm(f,1.10*self.Fmax,axes=1)
  new_vel=velocity+self.dt*(f/self.m-self.g*tf.constant([[0.,0.,1.]]))
  new_pos=position+self.dt/2.0*(velocity+new_vel)
  done=(new_pos[:,2]<0)
  return new_pos,new_vel,done
 def policy_loop(self,position,velocity,t,done,noise):
  new_pos,new_vel,done=self.step_multiple(position,velocity,t,noise)
  return new_pos,new_vel,t+1,done
 def policy_stop_condition(position,velocity,t,done):
  cond=tf.logical_not(done)
  cond=tf.logical_and(cond,t<SpacecraftEnv.MAXTIME)
  return cond
 def eager_while_loop(self,noise):
  batch_size=tf.shape(noise)[0]
  position=tf.convert_to_tensor(tf.tile([self.p0],[batch_size,1]))
  velocity=tf.convert_to_tensor(tf.tile([self.v0],[batch_size,1]))
  t_loop=tf.zeros([batch_size,],tf.int32)
  done_loop=tf.zeros([batch_size,],tf.bool)
  positions=[]
  velocities=[]
  positions.append(position)
  velocities.append(velocity)
  for index in range(SpacecraftEnv.MAXTIME):
   keepgoing=SpacecraftEnv.policy_stop_condition(position,velocity,t_loop,done_loop)
   if tf.reduce_all(tf.logical_not(keepgoing)):
    break
   position2,velocity2,t_loop2,done_loop2=self.policy_loop(position,velocity,t_loop,done_loop,tf.squeeze(noise[:,int(index/5.),:]))
   position=tf.where(tf.expand_dims(keepgoing,axis=1),position2,position)
   velocity=tf.where(tf.expand_dims(keepgoing,axis=1),velocity2,velocity)
   t_loop=tf.where(keepgoing,t_loop2,t_loop)
   done_loop=tf.where(keepgoing,done_loop2,done_loop)
   positions.append(position)
   velocities.append(velocity)
  return position,velocity,t_loop,done_loop,positions,velocities
def compute_obj_grad(noise,model_path):
 if not hasattr(compute_obj_grad,'env'):
  compute_obj_grad.env=SpacecraftEnv(optimal_file_path=model_path)
 tf.random.set_seed(123)
 batch_size=noise.shape[0]
 noise=tf.convert_to_tensor(noise,dtype=tf.float32)
 with tf.GradientTape()as g:
  g.watch(noise)
  out=compute_obj_grad.env.eager_while_loop(noise)
  cost=0.5*tf.norm(out[0],axis=1)**2
 dn=g.gradient(cost,noise)
 return cost.numpy().astype(np.float64),dn.numpy().astype(np.float64)
# Created by pyminifier (https://github.com/liftoff/pyminifier)

