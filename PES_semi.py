# first time reviewing the thing right now 
from cProfile import label
from distutils.command import clean
import os
import os.path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100

from tqdm import tqdm
from yaml import parse

from networks.ResNet import PreActResNet18
from common.tools import AverageMeter, getTime, evaluate, predict_softmax, train
from common.NoisyUtil import Train_Dataset, Semi_Labeled_Dataset, Semi_Unlabeled_Dataset, dataset_split
from selection import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=1, type=float, help='data number percent')
parser.add_argument('--noise_type', default='symmetric', type=str)
parser.add_argument('--noise_rate', default=0.5, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=5e-4)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--optim', default='cos', type=str, help='step, cos')

parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--lambda_u', default=150, type=float, help='weight for unsupervised loss')
parser.add_argument('--PES_lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--T1', default=0, type=int, help='if 0, set in below')
parser.add_argument('--T2', default=5, type=int, help='default 5')

parser.add_argument('--hyper_parameter_type', default=1, type=int, help='default to 1')
parser.add_argument('--modified', default=False, type=bool, help='default false')
parser.add_argument('--classifier', default='kmeans', type=str, help='default is kmeans')


args = parser.parse_args()
print(args)
os.system('nvidia-smi')


args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

gpu_id = 1

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True


def create_model(num_classes=10):
    model = PreActResNet18(num_classes)
    model.cuda(device=gpu_id)
    return model


def linear_rampup(current, warm_up=20, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

# Description of this algorithm:
# -> It takes labeled and unlabeled loader and other things that are being passed before 

# MixMatch Training
def MixMatch_train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights):
    net.train()
    if epoch >= args.num_epochs/2:
        args.alpha = 0.75

    losses = AverageMeter('Loss', ':6.2f')
    losses_lx = AverageMeter('Loss_Lx', ':6.2f')
    losses_lu = AverageMeter('Loss_Lu', ':6.5f')

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = int(50000/args.batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, targets_x = labeled_train_iter.next()

        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, inputs_x2, targets_x = inputs_x.cuda(device=gpu_id), inputs_x2.cuda(device=gpu_id), targets_x.cuda(device=gpu_id)
        inputs_u, inputs_u2 = inputs_u.cuda(device=gpu_id), inputs_u2.cuda(device=gpu_id)

        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
            ptu = pu**(1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixmatch_l = np.random.beta(args.alpha, args.alpha)
        mixmatch_l = max(mixmatch_l, 1 - mixmatch_l)

        mixed_input = mixmatch_l * input_a + (1 - mixmatch_l) * input_b
        mixed_target = mixmatch_l * target_a + (1 - mixmatch_l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx_mean = -torch.mean(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size * 2], 0)
        Lx = torch.sum(Lx_mean * class_weights)

        probs_u = torch.softmax(logits_u, dim=1)
        Lu = torch.mean((probs_u - mixed_target[batch_size * 2:])**2)
        loss = Lx + linear_rampup(epoch + batch_idx / num_iter, args.T1) * Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_lx.update(Lx.item(), batch_size * 2)
        losses_lu.update(Lu.item(), len(logits) - batch_size * 2)
        losses.update(loss.item(), len(logits))

    print(losses, losses_lx, losses_lu)

# gets two arrays of clean and noisy targets 
def splite_confident(outs, clean_targets, noisy_targets):
	probs, preds = torch.max(outs.data, 1) 

	confident_correct_num = 0
	unconfident_correct_num = 0
	confident_indexs = []
	unconfident_indexs = []
	
	# print("sonmethign is fos", len(clean_targets), len(noisy_targets))
	
	for i in range(0, len(noisy_targets)):
		if preds[i] == noisy_targets[i]:
			confident_indexs.append(i)
			if clean_targets[i] == preds[i]:
				confident_correct_num += 1
				
		else:
			unconfident_indexs.append(i)
			if clean_targets[i] == preds[i]:
				unconfident_correct_num += 1

	# print(getTime(), "confident and unconfident num:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2), len(unconfident_indexs), round(unconfident_correct_num / len(unconfident_indexs) * 100, 2)) 
	
	return confident_indexs, unconfident_indexs

def helperFunctionForFINE(train_data, noisy_targets, clean_labels):
	
    clean_idxs, preds = fine(train_data, noisy_targets, args.classifier, true_labels=clean_labels, args=args)

    # print("Length of the clean indexes here: " + str(len(clean_idxs)))
    
    clean_set = set(clean_idxs)
    noisy_idxs = []

    for idx in range(0, len(noisy_targets)):
        if idx not in clean_set:
            noisy_idxs.append(idx)

    return clean_idxs, noisy_idxs, preds

# FIXME: Pass enum type in future 
# FIXME: Try to come up with something similar afterwards
def return_confident_indexes(model, train_data, clean_targets, noisy_targets, isFine=False):

    if isFine:
        return helperFunctionForFINE(train_data, noisy_targets, clean_targets)

    else:

        predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        softouts = predict_softmax(predict_loader, model)
        return splite_confident(softouts, clean_targets, noisy_targets)

    return None, None


# takes the model, train data, clean targets, and noisy targets to return labeled, unlabeles loaders with class weights 

# it's literally just copy pasted from below
def update_train_loader_shiv(train_data, noisy_targets, confident_indexs, unconfident_indexs):
    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], transform_train)

    uncon_batch = int(args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * args.batch_size)
    if uncon_batch == 0: uncon_batch = 5
    con_batch = args.batch_size - uncon_batch

    # print(con_batch, uncon_batch)

    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[int(item)] += 1

    # TODO: might have to do this for FINE model as well (the class weights thing)

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss.
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda(device=gpu_id)
    # print("Category", train_nums, "precent", class_weights)
    return labeled_trainloader, unlabeled_trainloader, class_weights 

def update_trainloader(model, train_data, clean_targets, noisy_targets, isFine=False):

    # print("Update Train Loader is called")

    confident_indexs, unconfident_indexs = return_confident_indexes(model, train_data, clean_targets, noisy_targets, isFine)

    evaluate_accuracy(confident_indexs, unconfident_indexs, noisy_targets, clean_targets)

    confident_dataset = Semi_Labeled_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    unconfident_dataset = Semi_Unlabeled_Dataset(train_data[unconfident_indexs], transform_train)

    uncon_batch = int(args.batch_size / 2) if len(unconfident_indexs) > len(confident_indexs) else int(len(unconfident_indexs) / (len(confident_indexs) + len(unconfident_indexs)) * args.batch_size)
    if uncon_batch == 0: uncon_batch = 5
    con_batch = args.batch_size - uncon_batch

    # print(con_batch, uncon_batch)

    labeled_trainloader = DataLoader(dataset=confident_dataset, batch_size=con_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    unlabeled_trainloader = DataLoader(dataset=unconfident_dataset, batch_size=uncon_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:	
        train_nums[item] += 1

    # TODO: might have to do this for FINE model as well (the class weights thing)

    # zeros are not calculated by mean
    # avoid too large numbers that may result in out of range of loss.
    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0
        cw[cw > 3] = 3
    class_weights = torch.FloatTensor(cw).cuda(device=gpu_id)
    # print("Category", train_nums, "precent", class_weights)
    return labeled_trainloader, unlabeled_trainloader, class_weights


def noisy_refine(model, train_loader, num_layer, refine_times):
    if refine_times <= 0:
        return model
    # frezon all layers and add a new final layer
    for param in model.parameters():
        param.requires_grad = False

    model.renew_layers(num_layer)
    model.cuda(device=gpu_id)

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.PES_lr)
    for epoch in range(refine_times):
        train(model, train_loader, optimizer_adam, ceriation, epoch)
        _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")

    for param in model.parameters():
        param.requires_grad = True

    return model


if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':
    if args.T1 == 0:
        args.T1 = 20
    args.num_class = 10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':
    if args.T1 == 0:
        args.T1 = 35
    args.num_class = 100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)

if args.noise_type == "symmetric":
    noise_include = True
else:
    noise_include = False

ceriation = nn.CrossEntropyLoss().cuda(device=gpu_id)
data, _, noisy_labels, _, clean_labels, _ = dataset_split(train_set.data, np.array(train_set.targets), args.noise_rate, args.noise_type, args.data_percent, args.seed, args.num_class, noise_include)

if args.modified : 

    c = list(zip(data, noisy_labels, clean_labels))

    random.shuffle(c)

    data, noisy_labels, clean_labels = zip(*c)
    data, noisy_labels, clean_labels = np.asarray(data), np.asarray(noisy_labels), np.asarray(clean_labels)

train_dataset = Train_Dataset(data, noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)

model = create_model(num_classes=args.num_class)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
if args.optim == 'cos':
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
else:
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

best_test_acc = 0

# TODO: move all the code related to FINE to the selection folder 
# TODO: tidy up the code with comments and appropriate locations


'''
    # Command to run the code with flags and stuff 

    python PES_semi.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2  --lambda_u 15 --T1 5 --T2 5 --num_epochs 10 --modified True --classifier | tee trial_first_ever_run_5
'''

isFine = args.modified 

if isFine: 
    print("GMM Model used of type: " + str(ARGS.hyper_parameter_type))

print("Running with" + ("" if isFine else "out") + " FINE")
print("Epochs before Stopping: " + str(args.T1))
print("Epochs for reintialising: " + str(args.T2))
print("Epochs after Stopping: " + str(args.num_epochs - args.T1))

# features, labels = get_features(model, train_loader)
# quit()

def evaluate_accuracy(confi_idxs, unconfi_idxs, noisy_labels, clean_labels, train_noisy_labels=None):

    print("Size of labels: noisy -> " + str(noisy_labels.size) + " clean: " + str(clean_labels.size))
    print("Total: confident (" + str(len(confi_idxs)) + ") + (" + str(len(unconfi_idxs)) + ") = " + str(len(confi_idxs)+len(unconfi_idxs)))

    print("-" * 80)

    correct_confident_pred = 0

    for index in confi_idxs:
        if noisy_labels[index] == clean_labels[index]:
            correct_confident_pred += 1

    correct_unconfident_pred = 0

    for index in unconfi_idxs:
        if noisy_labels[index] != clean_labels[index]:
            correct_unconfident_pred += 1

    print("Confident: " + str(correct_confident_pred + correct_unconfident_pred) + " / " + str(noisy_labels.size))

    print("True detected:  " + str(correct_confident_pred))
    print("Noisy detected: " + str(correct_unconfident_pred))

    print("-" * 80)

    return 

# features, labels = get_features(model, train_loader)

# print("Features: ", features.shape)

# features_1 = get_features_custom(model, data)

# print("Features: ", features_1.shape)

# quit()

for epoch in range(args.num_epochs):

    print("Epoch: ", epoch)

    if epoch < args.T1:
        train(model, train_loader, optimizer, ceriation, epoch)
    else:
        if epoch == args.T1:
            model = noisy_refine(model, train_loader, 0, args.T2)

        true_labels = 0
        for idx in range(noisy_labels.size):
            if noisy_labels[idx] == clean_labels[idx]:
                true_labels += 1

        if isFine:
            # arguments required for mix match that update trainloader returns
            features, labels = get_features(model, train_loader)

            if epoch == args.T1:
                preds = np.copy(noisy_labels)
                preds = np.delete(preds, slice(49920, 50000))
                noisy_labels = np.delete(noisy_labels, slice(49920, 50000))

            # counter = 0

            # for idx in range(len(labels)):
            #     if labels[idx] == preds[idx]:
            #         counter += 1

            # print("Checking if the values could be updated: " + str(counter))

            # features_1 = get_features_custom(model, data)

            # print("Features: ", features_1.shape)

            # print("Labels shape: ", labels.shape) # 49920
            
            confident_indexs, unconfident_indexs, _ = helperFunctionForFINE(features, noisy_labels, clean_labels)

            evaluate_accuracy(confident_indexs, unconfident_indexs, noisy_labels, clean_labels)
            # print("Confident indexes shape: ", confident_indexs.shape)
            labeled_trainloader, unlabeled_trainloader, class_weights = update_train_loader_shiv(data, noisy_labels, confident_indexs, unconfident_indexs)

        else:

            labeled_trainloader, unlabeled_trainloader, class_weights = update_trainloader(model, data, clean_labels, noisy_labels, isFine)

        print("True label count: " + str(true_labels))

        print("_" * 80)

        # print("Length of the labeled trainloader: ", len(labeled_trainloader.dataset))
        # print("Length of the unlabeled trainloader: ", len(unlabeled_trainloader.dataset))

        # mixmatch to learn from the clean models and make the noisy models correct 
        MixMatch_train(epoch, model, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights)

    # validation 
    _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
    best_test_acc = test_acc if best_test_acc < test_acc else best_test_acc
    scheduler.step()

print(getTime(), "Best Test Acc:", best_test_acc)
