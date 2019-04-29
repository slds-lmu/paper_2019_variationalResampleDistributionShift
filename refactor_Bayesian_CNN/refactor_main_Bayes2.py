from __future__ import print_function

import os
import sys
import time
import argparse
import datetime
import math
import pickle
import numpy as np

import torchvision import torchvision.transforms as transforms

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import json

import Bayesian_config as cf


from utils.BBBlayers import GaussianVariationalInference

from utils.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from utils.BayesianModels.BayesianAlexNet import BBBAlexNet
from utils.BayesianModels.BayesianLeNet import BBBLeNet

import refactor_dataset_class
import utils_parent as utils_parent
from sklearn.model_selection import KFold

best_acc = 0


def getNetwork(args,inputs,outputs):
    if (args.net_type == 'lenet'):
        net = BBBLeNet(outputs,inputs)    # inputs is number of input channels
        file_name = 'lenet'
    elif (args.net_type == 'alexnet'):
        net = BBBAlexNet(outputs,inputs)
        file_name = 'alexnet-'
    elif (args.net_type == '3conv3fc'):
        net = BBB3Conv3FC(outputs,inputs)
        file_name = '3Conv3FC-'
    else:
        print('Error : Network should be either [LeNet / AlexNet /SqueezeNet/ 3Conv3FC')
        sys.exit(0)

    return net, file_name




# Training
def train(epoch,trainset,inputs,net,batch_size,trainloader,resize,num_epochs,use_cuda,vi,logfile):
    print("************")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    m = math.ceil(len(trainset) / batch_size)
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), weight_decay=args.weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        # repeat samples for
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        print(x.shape)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()  # GPU settings

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0
        # Forward Propagation
        x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)
        # print(outputs.shape)
        loss = vi(outputs, y, kl, beta)  # Loss
        optimizer.zero_grad()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (epoch, num_epochs, batch_idx + 1,
                                                                             (len(trainset) // batch_size) + 1,
                                                                             loss.data, (
                                                                                         100 * correct / total) / args.num_samples))
        sys.stdout.flush()

    # diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': (100*correct/total)/args.num_samples}
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': (100 * correct / total) / args.num_samples}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))
    return diagnostics_to_write


def test(epoch,testset,inputs,batch_size,testloader,net,use_cuda,num_epochs,resize,vi,logfile,file_name):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf = []
    m = math.ceil(len(testset) / batch_size)
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
        outputs, kl = net.probforward(x)

        if args.beta_type is "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif args.beta_type is "Soenderby":
            beta = min(epoch / (num_epochs // 4), 1)
        elif args.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        loss = vi(outputs, y, kl, beta)

        # test_loss += loss.data[0]
        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        preds = F.softmax(outputs, dim=1)
        results = torch.topk(preds.cpu().data, k=1, dim=1)
        conf.append(results[0][0].item())
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    p_hat = np.array(conf)
    confidence_mean = np.mean(p_hat, axis=0)
    confidence_var = np.var(p_hat, axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

    acc = (100 * correct / total) / args.num_samples
    # print('\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%' %(epoch, loss.data[0], acc))
    print('\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%' % (epoch, loss.data, acc))
    # test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data[0], 'Accuracy': acc}
    test_diagnostics_to_write = {'Validation Epoch': epoch, 'Loss': loss.data, 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
            'net': net if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/' + args.dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + file_name + '.t7')
        best_acc = acc
    return test_diagnostics_to_write



def prepare_data(args,train_list,test_list,resize):
    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

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

    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        outputs = 10
        inputs = 3

    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        outputs = 100
        inputs = 3

    elif (args.dataset == 'mnist'):
        print("| Preparing MNIST dataset...")
        sys.stdout.write("| ")
        # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        # testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
        trainset = refactor_dataset_class.VGMMDataset(list_idx = train_list,transform=transform_train)
        testset = refactor_dataset_class.VGMMDataset(list_idx = test_list,transform=transform_test)
        outputs = 10
        inputs = 1
    elif (args.dataset == 'fashion-mnist-random'):
        print("| Preparing fashion MNIST dataset for random cv...")
        sys.stdout.write("| ")
        # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        # testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
        trainset = refactor_dataset_class.VGMMDataset(index = train_list,transform=transform_train,cluster = False)
        testset = refactor_dataset_class.VGMMDataset(index = test_list,transform=transform_test,cluster =False)
        outputs = 10
        inputs = 1
    return trainset, testset, inputs,outputs

def prepare_data_for_normal_cv(args,train_list,test_list,resize):
    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

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

    if (args.dataset == 'mnist'):
        print("| Preparing fashion MNIST dataset for random cv...")
        sys.stdout.write("| ")
        # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        # testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
        trainset = refactor_dataset_class.VGMMDataset(index = train_list,transform=transform_train,cluster = False)
        testset = refactor_dataset_class.VGMMDataset(index = test_list,transform=transform_test,cluster =False)
        outputs = 10
        inputs = 1
    return trainset, testset, inputs,outputs

def cross_validation(num_labels,num_cluster,args):
    print("cross validation for random resampling")
    best_acc = 0
    resize = cf.resize
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    results = {}
    X, y = utils_parent.load_mnist('fashion-mnist')
    kf = KFold(n_splits=num_cluster)
    i = 0
    for train_eval_idx, test_idx in kf.split(X, y):
        i = i +1
    # for i in range(num_cluster):
    #     test_list = [i]
    #     train_list = list(range(num_cluster))
    #     train_list = [x for x in train_list if x != i]
    #     print(test_list, train_list)
        trainset, testset, inputs, outputs = prepare_data_for_normal_cv(args, train_eval_idx, test_idx, resize)
        # Hyper Parameter settings
        use_cuda = torch.cuda.is_available()
        use_cuda = cf.use_cuda()
        if use_cuda is True:
            torch.cuda.set_device(0)
        best_acc = 0
        resize = cf.resize
        start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)# Return network & file name

        # Model
        # Model
        print('\n[Phase 2] : Model setup')
        if args.resume:
            # Load checkpoint
            print('| Resuming from checkpoint...')
            assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
            _, file_name = getNetwork(args, inputs, outputs)
            checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
            net = checkpoint['net']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else:
            print('| Building net type [' + args.net_type + ']...')
            net, file_name = getNetwork(args, inputs, outputs)

        if use_cuda:
            net.cuda()

        vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

        #logfile = os.path.join('diagnostics_Bayes{}_{}.txt'.format(args.net_type, args.dataset))
        logfile_train = os.path.join('diagnostics_Bayes{}_{}_cv{}_train.txt'.format(args.net_type, args.dataset, i))
        logfile_test = os.path.join('diagnostics_Bayes{}_{}_cv{}_test.txt'.format(args.net_type, args.dataset, i))



        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(num_epochs))
        print('| Initial Learning Rate = ' + str(args.lr))
        print('| Optimizer = ' + str(optim_type))

        elapsed_time = 0

        train_return = []
        test_return = []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()

            temp_train_return = train(epoch, trainset, inputs, net, batch_size, trainloader, resize, num_epochs, use_cuda, vi, logfile_train)
            temp_test_return = test(epoch, testset, inputs, batch_size, testloader, net, use_cuda, num_epochs, resize, vi, logfile_test,
                 file_name)

            train_return = train_return.append(temp_train_return)
            test_return = test_return.append(temp_test_return)



            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

        print('\n[Phase 4] : Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))
        results[str(i)] = {"train": train_return, "test": test_return}
    return results


def cross_validation_for_clustered_data(num_labels,num_cluster,args):
    print("cross validation for clustered data")
    best_acc = 0
    resize = cf.resize
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    results = {}
    for i in range(num_cluster):
        test_list = [i]
        train_list = list(range(num_cluster))
        train_list = [x for x in train_list if x != i]
        print(test_list,train_list)
        trainset, testset,inputs,outputs = prepare_data(args,train_list,test_list,resize)
        # Hyper Parameter settings
        use_cuda = torch.cuda.is_available()
        use_cuda = cf.use_cuda()
        if use_cuda is True:
            torch.cuda.set_device(0)
        best_acc = 0
        resize = cf.resize
        start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)# Return network & file name

        # Model
        # Model
        print('\n[Phase 2] : Model setup')
        if args.resume:
            # Load checkpoint
            print('| Resuming from checkpoint...')
            assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
            _, file_name = getNetwork(args,inputs,outputs)
            checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
            net = checkpoint['net']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else:
            print('| Building net type [' + args.net_type + ']...')
            net, file_name = getNetwork(args,inputs,outputs)

        if use_cuda:
            net.cuda()

        vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())

        logfile_train = os.path.join('diagnostics_Bayes{}_{}_cv{}_train.txt'.format(args.net_type, args.dataset, i))
        logfile_test = os.path.join('diagnostics_Bayes{}_{}_cv{}_test.txt'.format(args.net_type, args.dataset, i))

        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(num_epochs))
        print('| Initial Learning Rate = ' + str(args.lr))
        print('| Optimizer = ' + str(optim_type))

        elapsed_time = 0
        train_return = []
        test_return = []
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()

            temp_train_return = train(epoch,trainset,inputs,net,batch_size,trainloader,resize,num_epochs,use_cuda,vi,logfile_train)
            temp_test_return = test(epoch,testset,inputs,batch_size,testloader,net,use_cuda,num_epochs,resize,vi,logfile_test,file_name)
            train_return = train_return.append(temp_train_return)
            test_return = test_return.append(temp_test_return)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

        print('\n[Phase 4] : Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))
        results[str(i)] = {"train": train_return, "test": test_return}

        # print('\n[Phase 3] : Training model')
        # print('| Training Epochs = ' + str(num_epochs))
        # print('| Initial Learning Rate = ' + str(args.lr))
        # print('| Optimizer = ' + str(optim_type))
        #
        # elapsed_time = 0
        # for epoch in range(start_epoch, start_epoch + num_epochs):
        #     start_time = time.time()
        #
        #     train(epoch,trainset,inputs,net,batch_size,trainloader,resize,num_epochs,use_cuda,vi,logfile)
        #     test(epoch,testset,inputs,batch_size,testloader,net,use_cuda,num_epochs,resize,vi,logfile,file_name)
        #
        #     epoch_time = time.time() - start_time
        #     elapsed_time += epoch_time
        #     print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
        #
        # print('\n[Phase 4] : Testing model')
        # print('* Test results : Acc@1 = %.2f%%' % (best_acc))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning_rate')
    parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
    # parser.add_argument('--depth', default=28, type=int, help='depth of model')
    # parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
    parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
    parser.add_argument('--p_logvar_init', default=0, type=int, help='p_logvar_init')
    parser.add_argument('--q_logvar_init', default=-10, type=int, help='q_logvar_init')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--cv_type', '-t', default = 'vgmm', type=str, help='cv_type=[rand/vgmm]')
    args = parser.parse_args()

    # cross_validation_for_clustered_data(num_labels=10,num_cluster=5,args=args)
    result = cross_validation(10,5,args)
    with open(args.cv_type + '_cross_validation_result.p', 'wb') as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.cv_type + '_cross_validation_result.json', 'w') as fp:
        json.dump(data, fp)





# cp rand_cross_validation_result.p
# cp rand_cross_validation_result.json









