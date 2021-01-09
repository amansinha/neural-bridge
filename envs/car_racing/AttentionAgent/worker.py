import argparse
import gin
import os
import misc.utility
import numpy as np
import shutil
import ZMQUtils
import zmq
def main(config):
 source_worker_socket,worker_sink_socket,source_worker_direct_socket,context= ZMQUtils.openSearchSocket(args.source_worker_port,args.worker_sink_port,args.source_worker_direct_port)
 source_worker_direct_socket.recv()
 source_worker_direct_socket.send(b"Ack")
 logger=misc.utility.create_logger(name='test_solution',debug=False)
 task=misc.utility.create_task(logger=logger)
 task.seed(config.seed)
 solution=misc.utility.create_solution()
 model_file=os.path.join(config.log_dir,config.model_filename)
 solution.load(model_file)
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
  task._env.env.noise=sample
  reward=task.roll_out(solution=solution,evaluate=True)
  temp=np.asarray([reward,index])
  ZMQUtils.send_array(worker_sink_socket,temp)
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--log-dir',help='Directory of logs.')
 parser.add_argument('--model-filename',default='model.npz',help='File name of the model to evaluate.')
 parser.add_argument('--render',help='Whether to render while evaluation.',default=False,action='store_true')
 parser.add_argument('--save-screens',help='Whether to save screenshots.',default=False,action='store_true')
 parser.add_argument('--overplot',help='Whether to render overplotted image.',default=False,action='store_true')
 parser.add_argument('--n-episodes',help='Number of episodes to evaluate.',type=int,default=3)
 parser.add_argument('--seed',help='Random seed for evaluation.',type=int,default=1)
 parser.add_argument('--source_worker_port',type=int)
 parser.add_argument('--worker_sink_port',type=int)
 parser.add_argument('--source_worker_direct_port',type=int)
 args,_=parser.parse_known_args()
 gin.parse_config_file(os.path.join(args.log_dir,'config.gin'))
 gin.bind_parameter("utility.create_task.render",args.render)
 if args.overplot:
  gin.bind_parameter("torch_solutions.VisionTaskSolution.show_overplot",True)
 main(args)
# Created by pyminifier (https://github.com/liftoff/pyminifier)

