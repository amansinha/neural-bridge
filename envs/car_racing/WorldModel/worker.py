import argparse
import numpy as np
import random
import json
import sys
from env import make_env
import time
from vae.vae import ConvVAE
from rnn.rnn import hps_sample,MDNRNN,rnn_init_state,rnn_next_state,rnn_output,rnn_output_size
import ZMQUtils
import zmq
import tensorflow as tf
render_mode=True
MODE_ZCH=0
MODE_ZC=1
MODE_Z=2
MODE_Z_HIDDEN=3 
MODE_ZH=4
EXP_MODE=MODE_ZH
def make_model(load_model=True):
 model=Model(load_model=load_model)
 return model
def sigmoid(x):
 return 1/(1+np.exp(-x))
def relu(x):
 return np.maximum(x,0)
def clip(x,lo=0.0,hi=1.0):
 return np.minimum(np.maximum(x,lo),hi)
def passthru(x):
 return x
def softmax(x):
 e_x=np.exp(x-np.max(x))
 return e_x/e_x.sum(axis=0)
def sample(p):
 return np.argmax(np.random.multinomial(1,p))
class Model:
 def __init__(self,load_model=True):
  self.env_name="carracing"
  self.vae=ConvVAE(batch_size=1,gpu_mode=False,is_training=False,reuse=True)
  self.rnn=MDNRNN(hps_sample,gpu_mode=False,reuse=True)
  if load_model:
   self.vae.load_json('vae/vae.json')
   self.rnn.load_json('rnn/rnn.json')
  self.state=rnn_init_state(self.rnn)
  self.rnn_mode=True
  self.input_size=rnn_output_size(EXP_MODE)
  self.z_size=32
  if EXP_MODE==MODE_Z_HIDDEN:
   self.hidden_size=40
   self.weight_hidden=np.random.randn(self.input_size,self.hidden_size)
   self.bias_hidden=np.random.randn(self.hidden_size)
   self.weight_output=np.random.randn(self.hidden_size,3)
   self.bias_output=np.random.randn(3)
   self.param_count=((self.input_size+1)*self.hidden_size)+(self.hidden_size*3+3)
  else:
   self.weight=np.random.randn(self.input_size,3)
   self.bias=np.random.randn(3)
   self.param_count=(self.input_size)*3+3
  self.render_mode=False
 def make_env(self,seed=-1,render_mode=False,full_episode=False):
  self.render_mode=render_mode
  self.env=make_env(self.env_name,seed=seed,render_mode=render_mode,full_episode=full_episode)
 def reset(self):
  self.state=rnn_init_state(self.rnn)
 def encode_obs(self,obs):
  result=np.copy(obs).astype(np.float)/255.0
  result=result.reshape(1,64,64,3)
  mu,logvar=self.vae.encode_mu_logvar(result)
  mu=mu[0]
  logvar=logvar[0]
  s=logvar.shape
  z=mu+np.exp(logvar/2.0)*np.random.randn(*s)
  return z,mu,logvar
 def get_action(self,z):
  h=rnn_output(self.state,z,EXP_MODE)
  if EXP_MODE==MODE_Z_HIDDEN:
   h=np.tanh(np.dot(h,self.weight_hidden)+self.bias_hidden)
   action=np.tanh(np.dot(h,self.weight_output)+self.bias_output)
  else:
   action=np.tanh(np.dot(h,self.weight)+self.bias)
  action[1]=(action[1]+1.0)/2.0
  action[2]=clip(action[2])
  self.state=rnn_next_state(self.rnn,z,action,self.state)
  return action
 def set_model_params(self,model_params):
  if EXP_MODE==MODE_Z_HIDDEN:
   params=np.array(model_params)
   cut_off=(self.input_size+1)*self.hidden_size
   params_1=params[:cut_off]
   params_2=params[cut_off:]
   self.bias_hidden=params_1[:self.hidden_size]
   self.weight_hidden=params_1[self.hidden_size:].reshape(self.input_size,self.hidden_size)
   self.bias_output=params_2[:3]
   self.weight_output=params_2[3:].reshape(self.hidden_size,3)
  else:
   self.bias=np.array(model_params[:3])
   self.weight=np.array(model_params[3:]).reshape(self.input_size,3)
 def load_model(self,filename):
  with open(filename)as f: 
   data=json.load(f)
  print('loading file %s'%(filename))
  self.data=data
  model_params=np.array(data[0])
  self.set_model_params(model_params)
 def get_random_model_params(self,stdev=0.1):
  return np.random.standard_cauchy(self.param_count)*stdev 
 def init_random_model_params(self,stdev=0.1):
  params=self.get_random_model_params(stdev=stdev)
  self.set_model_params(params)
  vae_params=self.vae.get_random_model_params(stdev=stdev)
  self.vae.set_model_params(vae_params)
  rnn_params=self.rnn.get_random_model_params(stdev=stdev)
  self.rnn.set_model_params(rnn_params)
def simulate(model,train_mode=False,render_mode=True,num_episode=5,seed=-1,max_len=-1):
 reward_list=[]
 t_list=[]
 max_episode_length=1000
 recording_mode=False
 penalize_turning=False
 if train_mode and max_len>0:
  max_episode_length=max_len
 if(seed>=0):
  random.seed(seed)
  np.random.seed(seed)
  model.env.seed(seed)
 for episode in range(num_episode):
  model.reset()
  obs=model.env.reset()
  if obs is None:
   return np.inf,np.inf
  total_reward=0.0
  random_generated_int=np.random.randint(2**31-1)
  filename="record/"+str(random_generated_int)+".npz"
  recording_mu=[]
  recording_logvar=[]
  recording_action=[]
  recording_reward=[0]
  for t in range(max_episode_length):
   if render_mode:
    model.env.render("human")
   else:
    model.env.render('rgb_array')
   z,mu,logvar=model.encode_obs(obs)
   action=model.get_action(z)
   recording_mu.append(mu)
   recording_logvar.append(logvar)
   recording_action.append(action)
   obs,reward,done,info=model.env.step(action)
   extra_reward=0.0 
   if train_mode and penalize_turning:
    extra_reward-=np.abs(action[0])/10.0
    reward+=extra_reward
   recording_reward.append(reward)
   total_reward+=reward
   if done:
    break
  z,mu,logvar=model.encode_obs(obs)
  action=model.get_action(z)
  recording_mu.append(mu)
  recording_logvar.append(logvar)
  recording_action.append(action)
  recording_mu=np.array(recording_mu,dtype=np.float16)
  recording_logvar=np.array(recording_logvar,dtype=np.float16)
  recording_action=np.array(recording_action,dtype=np.float16)
  recording_reward=np.array(recording_reward,dtype=np.float16)
  if not render_mode:
   if recording_mode:
    np.savez_compressed(filename,mu=recording_mu,logvar=recording_logvar,action=recording_action,reward=recording_reward)
  if render_mode:
   print("total reward",total_reward,"timesteps",t)
  reward_list.append(total_reward)
  t_list.append(t)
 return total_reward,t_list
parser=argparse.ArgumentParser()
parser.add_argument('--source_worker_port',type=int)
parser.add_argument('--worker_sink_port',type=int)
parser.add_argument('--source_worker_direct_port',type=int)
args=parser.parse_args()
render_mode=False
filename='log/carracing.cma.16.64.best.json'
the_seed=123 
np.random.seed(the_seed)
model=make_model()
print('model size',model.param_count)
model.make_env(render_mode=render_mode,seed=the_seed)
model.load_model(filename)
source_worker_socket,worker_sink_socket,source_worker_direct_socket,context=ZMQUtils.openSearchSocket(args.source_worker_port,args.worker_sink_port,args.source_worker_direct_port)
source_worker_direct_socket.recv()
source_worker_direct_socket.send(b"Ack")
while True:
 try:
  s=source_worker_direct_socket.recv(flags=zmq.NOBLOCK)
  source_worker_direct_socket.send(b"Ack")
  assert(s==b"end")
  break
 except zmq.Again as e:
  pass
 try:
  wholething=ZMQUtils.recv_array(source_worker_socket,flags=zmq.NOBLOCK)
 except zmq.Again as e:
  continue
 sample=wholething[:-1].reshape((12,2))
 index=wholething[-1]
 model.env.noise=sample
 np.random.seed(the_seed)
 tf.set_random_seed(the_seed)
 reward,steps_taken=simulate(model,train_mode=False,seed=the_seed,render_mode=render_mode,num_episode=1)
 temp=np.asarray([reward,index])
 ZMQUtils.send_array(worker_sink_socket,temp)
# Created by pyminifier (https://github.com/liftoff/pyminifier)

