import numpy as np
import pickle
import scipy.stats as sps
import ZMQUtils
def compute_objective(sample,index,socket):
 temp=sps.norm.cdf(sample)
 wholething=np.concatenate((temp,np.array([index])))
 ZMQUtils.send_array(socket,np.ascontiguousarray(wholething))
def _compute_objectives(X,source_sink_req_socket,source_worker_socket):
 source_sink_req_socket.send(str(len(X)).encode())
 source_sink_req_socket.recv()
 for i in range(len(X)):
  compute_objective(X[i],i,source_worker_socket)
 source_sink_req_socket.send(b"done")
 temp=-1*ZMQUtils.recv_array(source_sink_req_socket)
 return temp
def compute_objectives(X):
 return _compute_objectives(X,source_sink_req_socket,source_worker_socket)
parser=ZMQUtils.Args_makeStandardParser()
parser.add_argument('--num_samples',type=int,default=10)
args=parser.parse_args()
ZMQUtils.Args_convertStandard(args)
np.random.seed(12345+args.startport)
context,source_worker_socket,source_sink_req_socket,sink_process,source_worker_direct_sockets= ZMQUtils.Args_getSockets(args)
print('Ready to start naiving')
X=np.random.randn(args.num_samples,24)
Xobj=compute_objectives(X)
np.savez_compressed('naive'+str(args.startport)+'.npz',X=X,Xobj=Xobj)
ZMQUtils.heartbeat(source_worker_direct_sockets,b"end")
sink_process.terminate()
context.destroy()
# Created by pyminifier (https://github.com/liftoff/pyminifier)

