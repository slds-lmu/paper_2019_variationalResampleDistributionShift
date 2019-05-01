from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import Frequentist_config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import numpy as np

from torch.autograd import Variable


from utils.FrequentistModels import conv_init
from utils.FrequentistModels.AlexNet import AlexNet
from utils.FrequentistModels.LeNet import LeNet
from utils.FrequentistModels.F3Conv3FC import F3Conv3FC
import refactor_dataset_class
import utils_parent as utils_parent
from sklearn.model_selection import KFold
import config as config_parent
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

# Return network & file name
def getNetwork(args,inputs,outputs):
    if (args.net_type == 'lenet'):
        net = LeNet(outputs,inputs)
        file_name = 'lenet'
    elif (args.net_type == 'alexnet'):
        net = AlexNet(outputs,inputs)
        file_name = 'alexnet-'
    elif (args.net_type == '3conv3fc'):
        net = F3Conv3FC(outputs, inputs)
        file_name = 'ThreeConvThreeFC-'
    else:
        print('Error : Network should be either [LeNet / AlexNet /SqueezeNet/ ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# # Test only option
# if (args.testOnly):
#     print('\n[Test Phase] : Model setup')
#     assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
#     _, file_name = getNetwork(args)
#     checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
#     net = checkpoint['net']
#
#     if use_cuda:
#         net.cuda()
#         net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#         cudnn.benchmark = True
#
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     for batch_idx, (inputs, targets) in enumerate(testloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs = net(inputs)
#
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()
#
#     acc = 100.*correct/total
#     print("| Test Result\tAcc@1: %.2f%%" %(acc))
#
#     sys.exit(0)



# Training
def train(epoch,trainset,inputs,net,batch_size,trainloader,resize,num_epochs,use_cuda,criterion,logfile):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_value, targets) in enumerate(trainloader):
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()  # GPU settings
            # inputs_value, targets = inputs_value.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        # Forward Propagation
        x, y = Variable(x), Variable(y)
        # inputs_value, targets = Variable(inputs_value), Variable(targets)
        # outputs = net(inputs_value)               # Forward Propagation
        outputs = net(x)               # Forward Propagation
        # loss = criterion(outputs, targets)  # Loss
        loss = criterion(outputs, y)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.data, (100. * correct / total) / args.num_samples))
        sys.stdout.flush()
    diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': (100. * correct / total) / args.num_samples}
    with open(logfile, 'a') as lf:
        lf.write(str(diagnostics_to_write))
    return diagnostics_to_write

def test(epoch,testset,inputs,batch_size,testloader,net,use_cuda,num_epochs,resize,criterion,logfile,file_name):
    global best_acc
    best_acc = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs_value, targets) in enumerate(testloader):
        # x = inputs_value.repeat(args.num_samples, 1, 1, 1)
        x = inputs_value.view(-1, inputs, resize, resize).repeat(args.num_samples, 1, 1, 1)
        y = targets.repeat(args.num_samples)
        if use_cuda:
            x, y = x.cuda(), y.cuda()
            # inputs_value, targets = inputs_value.cuda(), targets.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y)
            # inputs_value, targets = Variable(inputs_value), Variable(targets)
        # outputs = net(inputs_value)
        outputs = net(x)
        # loss = criterion(outputs, targets)
        loss = criterion(outputs, y)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

    # Save checkpoint when best model
    # acc = 100.*correct/total
    acc = (100 * correct / total) / args.num_samples
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data, acc))
    test_diagnostics_to_write = {'Epoch': epoch, 'Loss': loss.data, 'Accuracy': acc}
    with open(logfile, 'a') as lf:
        lf.write(str(test_diagnostics_to_write))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc
    return test_diagnostics_to_write


def prepare_data(args,train_eval_list,test_list,resize):
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
        # num_classe = 10
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

        if args.debug ==True:
            train_eval_set = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                                root_dir="../" + config_parent.data_path,
                                                                list_idx=train_eval_list, transform=transform_train)
            # only get subset of original dataset
            small_size = int(0.01*len(train_eval_set))
            drop_size = len(train_eval_set)-small_size
            train_eval_set,_ = torch.utils.data.random_split(train_eval_set, [small_size, drop_size])

            # split train_eval_set into trainset and evalset
            train_size = int(0.8 * len(train_eval_set))
            eval_size = len(train_eval_set) - train_size
            trainset, evalset = torch.utils.data.random_split(train_eval_set, [train_size, eval_size])
            testset = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                         root_dir="../" + config_parent.data_path, list_idx=test_list,
                                                         transform=transform_test)
            small_size = int(0.01 * len(testset))
            drop_size = len(testset) - small_size
            testset,_ =torch.utils.data.random_split(testset, [small_size, drop_size])
            outputs = 10
            inputs = 1
        else:

            train_eval_set = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                                root_dir="../" + config_parent.data_path,
                                                                list_idx=train_eval_list, transform=transform_train)
            # split train_eval_set into trainset and evalset
            train_size = int(0.8 * len(train_eval_set))
            eval_size = len(train_eval_set) - train_size
            trainset, evalset = torch.utils.data.random_split(train_eval_set, [train_size, eval_size])
            testset = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                         root_dir="../" + config_parent.data_path, list_idx=test_list,
                                                         transform=transform_test)
            outputs = 10
            inputs = 1

    return trainset, evalset, testset, inputs, outputs


def prepare_data_for_normal_cv(args,train_eval_list,test_list,resize):
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
        if args.debug == True:
            train_eval_set = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                                root_dir="../" + config_parent.data_path,
                                                                index=train_eval_list, transform=transform_train,
                                                                cluster=False)
            # only get subset of original dataset
            small_size = int(0.01*len(train_eval_set))
            drop_size = len(train_eval_set)-small_size
            train_eval_set,_ = torch.utils.data.random_split(train_eval_set, [small_size, drop_size])

            # split train_eval_set into trainset and evalset
            train_size = int(0.8 * len(train_eval_set))
            eval_size = len(train_eval_set) - train_size
            trainset, evalset = torch.utils.data.random_split(train_eval_set, [train_size, eval_size])
            testset = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                         root_dir="../" + config_parent.data_path, index=test_list,
                                                         transform=transform_test, cluster=False)
            small_size = int(0.01 * len(testset))
            drop_size = len(testset) - small_size
            testset, _ = torch.utils.data.random_split(testset, [small_size, drop_size])
            outputs = 10
            inputs = 1
        else:
            train_eval_set = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                                root_dir="../" + config_parent.data_path,
                                                                index=train_eval_list, transform=transform_train,
                                                                cluster=False)
            # split train_eval_set into trainset and evalset
            train_size = int(0.8 * len(train_eval_set))
            eval_size = len(train_eval_set) - train_size
            trainset, evalset = torch.utils.data.random_split(train_eval_set, [train_size, eval_size])
            testset = refactor_dataset_class.VGMMDataset(pattern=config_parent.global_index_name,
                                                         root_dir="../" + config_parent.data_path, index=test_list,
                                                         transform=transform_test, cluster=False)
            outputs = 10
            inputs = 1

    return trainset, evalset, testset, inputs,outputs


def cross_validation(num_labels,num_cluster,args):
    print("cross validation for random resampling")
    best_acc = 0
    # resize=32
    resize = cf.resize
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    results = {}
    X, y = utils_parent.load_mnist('fashion-mnist')
    kf = KFold(n_splits=num_cluster)
    i = 0
    for train_eval_idx, test_idx in kf.split(X, y):
        i = i +1
        trainset, evalset, testset, inputs, outputs = prepare_data_for_normal_cv(args, train_eval_idx, test_idx, resize)
        # Hyper Parameter settings
        use_cuda = torch.cuda.is_available()
        use_cuda = cf.use_cuda()
        if use_cuda is True:
            torch.cuda.set_device(0)
        best_acc = 0
        resize = cf.resize
        start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        evalloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            net.apply(conv_init)

        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        # logfile = os.path.join('diagnostics_NonBayes{}_{}.txt'.format(args.net_type, args.dataset))
        logfile_train = os.path.join('diagnostics_NonBayes{}_{}_cv{}_train.txt'.format(args.net_type, args.dataset, i))
        logfile_test = os.path.join('diagnostics_NonBayes{}_{}_cv{}_test.txt'.format(args.net_type, args.dataset, i))

        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(num_epochs))
        print('| Initial Learning Rate = ' + str(args.lr))
        print('| Optimizer = ' + str(optim_type))

        elapsed_time = 0

        train_return = []
        test_return = []
        eval_return = []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()

            # train(epoch)
            # test(epoch)
            temp_train_return = train(epoch, trainset, inputs, net, batch_size, trainloader, resize, num_epochs, use_cuda, criterion, logfile_train)
            temp_eval_return = test(epoch, evalset, inputs, batch_size, evalloader, net, use_cuda, num_epochs, resize, criterion, logfile_test,file_name)

            # train_return = train_return.append(temp_train_return)
            train_return = np.append(train_return,temp_train_return)
            # eval_return = eval_return.append(temp_eval_return)
            eval_return = np.append(eval_return,temp_eval_return)

            temp_test_return = test(epoch, testset, inputs, batch_size, testloader, net, use_cuda, num_epochs, resize,
                                    criterion, logfile_test, file_name)
            # test_return = test_return.append(temp_test_return)
            test_return = np.append(test_return, temp_test_return)
            print(temp_train_return)
            print(temp_eval_return)
            print(temp_test_return)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

        print('\n[Phase 4] : Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))
        results[str(i)] = {"train": train_return, "test": test_return, "eval": eval_return}
        print(results)
    return results



def cross_validation_for_clustered_data(num_labels,num_cluster,args):
    print("cross validation for clustered resampling")
    best_acc = 0
    # resize=32
    resize = cf.resize
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    results = {}
    for i in range(num_cluster):
        test_list = [i]
        train_eval_list = list(range(num_cluster))
        train_eval_list = [x for x in train_eval_list if x != i]
        print(test_list,train_eval_list)
        trainset, evalset, testset,inputs,outputs = prepare_data(args,train_eval_list,test_list,resize)
        # Hyper Parameter settings
        # use_cuda = torch.cuda.is_available()
        use_cuda = cf.use_cuda()
        if use_cuda is True:
            torch.cuda.set_device(0)
        best_acc = 0
        resize = cf.resize
        start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            net.apply(conv_init)

        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        # logfile = os.path.join('diagnostics_NonBayes{}_{}.txt'.format(args.net_type, args.dataset))
        logfile_train = os.path.join('diagnostics_NonBayes{}_{}_cv{}_train.txt'.format(args.net_type, args.dataset, i))
        logfile_test = os.path.join('diagnostics_NonBayes{}_{}_cv{}_test.txt'.format(args.net_type, args.dataset, i))

        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(num_epochs))
        print('| Initial Learning Rate = ' + str(args.lr))
        print('| Optimizer = ' + str(optim_type))

        elapsed_time = 0
        train_return = []
        eval_return = []
        test_return = []
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()

            # train(epoch)
            # test(epoch)
            temp_train_return = train(epoch,trainset,inputs,net,batch_size,trainloader,resize,num_epochs,use_cuda,criterion,logfile_train)
            temp_eval_return = test(epoch,evalset,inputs,batch_size,evalloader,net,use_cuda,num_epochs,resize,criterion,logfile_test,file_name)
            # train_return = train_return.append(temp_train_return)
            train_return = np.append(train_return,temp_train_return)
            # eval_return = eval_return.append(temp_eval_return)
            eval_return = np.append(train_return,temp_eval_return)


            temp_test_return = test(epoch, testset, inputs, batch_size, testloader, net, use_cuda, num_epochs, resize,
                                    criterion,
                                    logfile_test, file_name)

            # test_return = test_return.append(temp_test_return)
            test_return = np.append(test_return, temp_test_return)
            print(temp_train_return)
            print(temp_eval_return)
            print(temp_test_return)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

        print('\n[Phase 4] : Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))
        results[str(i)] = {"train": train_return, "test": test_return,"val":eval_return}
        print(results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
    parser.add_argument('--net_type', default='alexnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='dataset = [mnist/cifar10/cifar100/fashionmnist/stl10]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--cv_type', '-v', default='vgmm', type=str, help='cv_type=[rand/vgmm]')
    parser.add_argument('--debug', default=False, type=bool, help="debug mode has smaller data")
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
    args = parser.parse_args()


    if args.cv_type == "vgmm":
        result = cross_validation_for_clustered_data(num_labels=config_parent.num_labels,num_cluster=config_parent.num_clusters,args=args)
    else:
        result = cross_validation(config_parent.num_labels,config_parent.num_clusters,args)
    with open(args.cv_type + '_cross_validation_result.p', 'wb') as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # cause ndarray is not json serializable
    # with open(args.cv_type + '_cross_validation_result.json', 'w') as fp:
    #     json.dump(result, fp)

    np.save(args.cv_type+'_cross_validation_result.npy',result)
    utils_parent.write_results_to_csv(args.cv_type+'_cross_validation_result.csv',result)