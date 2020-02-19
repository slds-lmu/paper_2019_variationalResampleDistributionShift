# check the documentation here: https://pytorch.org/docs/stable/notes/multiprocessing.html
# check the documentation here: https://docs.python.org/3/library/multiprocessing.html
import torch.multiprocessing as torch_multiprocessing
import multiprocessing as org_multiprocessing
from torch.multiprocessing.pool import Pool as TorchPool
from multiprocessing.pool import Pool as OrgPool
#torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module. It registers custom reducers, that use shared memory to provide shared views on the same data in different processes. Once the tensor/storage is movedto shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible to send it to other processes without making any copies.
import os


torch_multiprocessing.set_start_method('spawn', force = True) #The CUDA runtime does not support the fork start method.


class OrgNoDaemonProcess(org_multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)



class TorchNoDaemonProcess(torch_multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyOrgPool(OrgPool):
    Process = OrgNoDaemonProcess

class MyTorchPool(OrgPool):  # always has to subclass OrgPool instead of TorchPool due to bug in pytorch
    Process = OrgNoDaemonProcess



def f(x):
    print(x)
    return x*x

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def ff(name):
    info('function f')
    print('hello', name)

#lambda a : a + 10
with MyOrgPool(torch_multiprocessing.cpu_count()) as p:
    result = p.map(f,list(range(5)))
result
with MyTorchPool(torch_multiprocessing.cpu_count()) as p:
    result = p.map(f,list(range(5)))
result



processes = []
i = 0.0
for rank in range(3):
    i = i +1
    p = torch_multiprocessing.Process(target=ff, args=('hallo',))
    p.start()
    processes.append(p)

for rank in range(3):
    i = i +1
    p = torch_multiprocessing.Process(target=f, args=(i,))
    p.start()
    processes.append(p)



for p in processes:
    p.join()


