import torchvision
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import refactor_dataset_class
import Bayesian_config as cf
resize = cf.resize

import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning_rate')
parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
#parser.add_argument('--depth', default=28, type=int, help='depth of model')
#parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
#args = parser.parse_args()
args = parser.parse_args(["--dataset", "mnist"])
args


transform_train = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])  # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
trainset_org = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainset_refactor = refactor_dataset_class.VGMMDataset()
trainloader_org = torch.utils.data.DataLoader(trainset_org, batch_size=batch_size, shuffle=True, num_workers=4)
trainloader_refactor = torch.utils.data.DataLoader(trainset_refactor, batch_size=batch_size, shuffle=False, num_workers=4)
# num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)# Return network & file name
import utils_parent
X, y = utils_parent.load_mnist("fashion-mnist")
type(y)
y.shape
ynew = y.argmax(axis = 1)
ynew
type(ynew)

for batch_idx, (inputs_value, targets) in enumerate(trainloader_org):
    print(inputs_value)
    print(targets)
    print("......")
    print(targets.type)
    print(targets.shape)
    break

for batch_idx, (inputs_value, targets) in enumerate(trainloader_refactor):
    print(inputs_value)
    print(targets)
    print("......")
    print(targets.type)
    print(targets.shape)
    break
