import argparse
import multiprocessing
import numpy as np
import os
import time
import zmq
class cd:
 def __init__(self,newPath):
  self.newPath=os.path.expanduser(newPath)
 def __enter__(self):
  self.savedPath=os.getcwd()
  os.chdir(self.newPath)
 def __exit__(self,etype,value,traceback):
  os.chdir(self.savedPath)
def send_array(socket,A,flags=0,copy=True,track=False):
 md=dict(dtype=str(A.dtype),shape=A.shape)
 socket.send_json(md,flags|zmq.SNDMORE)
 return socket.send(A,flags,copy=copy,track=track)
def recv_array(socket,flags=0,copy=True,track=False):
 md=socket.recv_json(flags=flags)
 msg=socket.recv(flags=flags,copy=copy,track=track)
 buf=memoryview(msg)
 A=np.frombuffer(buf,dtype=md['dtype'])
 return A.reshape(md['shape'])
def heartbeat(source_worker_direct_sockets,msg):
 for socket in source_worker_direct_sockets:
  socket.send(msg)
  socket.recv()
def openSearchSocket(source_worker_port,worker_sink_port,source_worker_direct_port):
 context=zmq.Context()
 source_worker_socket=context.socket(zmq.PULL)
 source_worker_socket.connect("tcp://localhost:"+str(source_worker_port))
 worker_sink_socket=context.socket(zmq.PUSH)
 worker_sink_socket.connect("tcp://localhost:"+str(worker_sink_port))
 source_worker_direct_socket=context.socket(zmq.REP)
 source_worker_direct_socket.bind("tcp://*:"+str(source_worker_direct_port))
 return source_worker_socket,worker_sink_socket,source_worker_direct_socket,context
def sink_thread(worker_sink_port,source_sink_port):
 context=zmq.Context()
 worker_sink_socket=context.socket(zmq.PULL)
 worker_sink_socket.bind("tcp://*:"+str(worker_sink_port))
 source_sink_rep_socket=context.socket(zmq.REP)
 source_sink_rep_socket.bind("tcp://*:"+str(source_sink_port))
 print('Sink ready to go')
 while True:
  num_simult_runs=int(source_sink_rep_socket.recv())
  source_sink_rep_socket.send(b" ")
  objectives=np.empty((num_simult_runs,))
  for simult_run in range(num_simult_runs):
   stuff=recv_array(worker_sink_socket)
   objective=stuff[0]
   index=int(stuff[1])
   objectives[index]=objective
  s=source_sink_rep_socket.recv()
  assert(s==b"done")
  send_array(source_sink_rep_socket,objectives)
 context.destroy()
def Args_makeStandardParser():
 parser=argparse.ArgumentParser()
 parser.add_argument('--startport',type=int,default=5000)
 parser.add_argument('--num_workers',type=int,default=1)
 return parser
def Args_convertStandard(args):
 pass
def Args_getSockets(args):
 startport=args.startport
 num_workers=args.num_workers
 source_worker_port=startport
 worker_sink_port=startport+1
 source_sink_port=startport+2
 source_worker_direct_port=startport+3
 print('Ports:')
 print('source_worker_port',source_worker_port)
 print('worker_sink_port',worker_sink_port)
 print('source_sink_port',source_sink_port)
 print('source_worker_direct_ports',source_worker_direct_port,'-',source_worker_direct_port+num_workers-1)
 print('')
 context=zmq.Context()
 source_worker_socket=context.socket(zmq.PUSH)
 source_worker_socket.bind("tcp://*:"+str(source_worker_port))
 source_sink_req_socket=context.socket(zmq.REQ)
 source_sink_req_socket.connect("tcp://localhost:"+str(source_sink_port))
 sink_process=multiprocessing.Process(target=sink_thread,args=(worker_sink_port,source_sink_port))
 sink_process.start()
 time.sleep(2)
 source_worker_direct_sockets=[]
 for i in range(num_workers):
  socket=context.socket(zmq.REQ)
  print('Creating source-worker direct port',source_worker_direct_port+i)
  socket.connect("tcp://localhost:"+str(source_worker_direct_port+i))
  source_worker_direct_sockets.append(socket)
  os.system('xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" python3 worker.py'+' --source_worker_port '+str(source_worker_port)+' --worker_sink_port '+str(worker_sink_port)+' --source_worker_direct_port '+str(source_worker_direct_port+i)+' 1>&2 &')
  time.sleep(0.5)
 for i in range(len(source_worker_direct_sockets)):
  source_worker_direct_sockets[i].send(b" ")
  source_worker_direct_sockets[i].recv()
  print('Worker ',i,' ready to go')
 return context,source_worker_socket,source_sink_req_socket,sink_process,source_worker_direct_sockets
# Created by pyminifier (https://github.com/liftoff/pyminifier)

