import tensorflow as tf
import numpy as np
from tensorflow.keras import models
class TFMountainCarEnv(object):
 MAXTIME=1000
 def __init__(self,min_action=-1.0,max_action=1.0,min_position=-1.2,max_position=0.6,max_speed=0.07,goal_position=0.45,goal_velocity=0.,power=0.0015,agent_model_path=None):
  self.min_action=min_action
  self.max_action=max_action
  self.min_position=min_position
  self.max_position=max_position
  self.max_speed=max_speed
  self.goal_position=goal_position
  self.goal_velocity=goal_velocity
  self.power=power
  self.low_state=np.array([self.min_position,-self.max_speed],dtype=np.float32)
  self.high_state=np.array([self.max_position,self.max_speed],dtype=np.float32)
  if agent_model_path is not None:
   self.agent=models.load_model(agent_model_path)
 def step(self,state,action):
  position=state[0]
  velocity=state[1]
  force=tf.clip_by_value(action[0],self.min_action,self.max_action)
  velocity+=force*self.power-0.0025*tf.cos(3*position)
  velocity=tf.clip_by_value(velocity,-1*self.max_speed,self.max_speed)
  position+=velocity
  position=tf.clip_by_value(position,self.min_position,self.max_position)
  velocity=tf.cond(tf.logical_and(tf.abs(position-self.min_position)<1e-6,velocity<0),lambda:0*velocity,lambda:velocity)
  done=tf.logical_and(position>=self.goal_position,velocity>=self.goal_velocity)
  reward=-0.1*tf.pow(action[0],2)
  reward=tf.cond(done,lambda:reward+100,lambda:reward)
  return tf.stack([position,velocity]),reward,done
 def step_multiple(self,state,action):
  position=state[:,0]
  velocity=state[:,1]
  force=tf.clip_by_value(action,self.min_action,self.max_action)
  velocity+=force*self.power-0.0025*tf.cos(3*position)
  velocity=tf.clip_by_value(velocity,-1*self.max_speed,self.max_speed)
  position+=velocity
  position=tf.clip_by_value(position,self.min_position,self.max_position)
  velocity=tf.where(tf.logical_and(tf.abs(position-self.min_position)<1e-6,velocity<0),0*velocity,velocity)
  done=tf.logical_and(position>=self.goal_position,velocity>=self.goal_velocity)
  reward=-0.1*tf.pow(action,2)
  reward=tf.where(done,reward+100,reward)
  return tf.stack([position,velocity],axis=1),reward,done
 def policy_loop(self,state_,t,totalreward,done):
  action=tf.squeeze(self.agent(state_),axis=1)
  state,reward,done=self.step_multiple(state_,action)
  totalreward+=reward
  return state,t+1,totalreward,done
 def policy_stop_condition(state_,t,totalreward,done):
  cond=tf.logical_not(done)
  cond=tf.logical_and(cond,t<TFMountainCarEnv.MAXTIME)
  return cond
 def eager_while_loop(self,states):
  states_loop=states
  t_loop=tf.zeros([tf.shape(states)[0],],tf.int32)
  totalreward_loop=tf.zeros([tf.shape(states)[0],],tf.float32)
  done_loop=tf.zeros([tf.shape(states)[0],],tf.bool)
  rollout=[]
  rollout.append(states_loop)
  for _ in range(TFMountainCarEnv.MAXTIME):
   keepgoing=TFMountainCarEnv.policy_stop_condition(states_loop,t_loop,totalreward_loop,done_loop)
   if tf.reduce_all(tf.logical_not(keepgoing)):
    break
   states_loop2,t_loop2,totalreward_loop2,done_loop2=self.policy_loop(states_loop,t_loop,totalreward_loop,done_loop)
   states_loop=tf.where(tf.expand_dims(keepgoing,axis=1),states_loop2,states_loop)
   t_loop=tf.where(keepgoing,t_loop2,t_loop)
   totalreward_loop=tf.where(keepgoing,totalreward_loop2,totalreward_loop)
   done_loop=tf.where(keepgoing,done_loop2,done_loop)
   rollout.append(states_loop)
  return states_loop,t_loop,totalreward_loop,done_loop,rollout
def compute_obj_grad(position,velocity,model_path):
 if not hasattr(compute_obj_grad,'env'):
  compute_obj_grad.env=TFMountainCarEnv(agent_model_path=model_path)
 tf.random.set_seed(123)
 position=tf.convert_to_tensor(position,dtype=tf.float32)
 velocity=tf.convert_to_tensor(velocity,dtype=tf.float32)
 with tf.GradientTape(persistent=True)as g:
  g.watch(position)
  g.watch(velocity)
  states=tf.stack([position,velocity],axis=1)
  out=compute_obj_grad.env.eager_while_loop(states)
 dp=g.gradient(out[2],position)
 dv=g.gradient(out[2],velocity)
 del g
 return out[2].numpy().astype(np.float64),np.stack([dp,dv],axis=1).astype(np.float64)
# Created by pyminifier (https://github.com/liftoff/pyminifier)

