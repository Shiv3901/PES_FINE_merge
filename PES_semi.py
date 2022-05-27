# first time reviewing the thing right now 
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

# try to locate where this function is being called and it is possible to see where we can fit FINE in 

# gets two arrays of clean and noisy targets 
def splite_confident(outs, clean_targets, noisy_targets):
	probs, preds = torch.max(outs.data, 1) 

	confident_correct_num = 0
	unconfident_correct_num = 0
	confident_indexs = []
	unconfident_indexs = []
	
	print("sonmethign is fos", len(clean_targets), len(noisy_targets))
	
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

def helperFunctionForFINE(train_data, noisy_targets, k=100):
	
    # FIXME: consider making a set for this one as it could be possible that you are counting it more than once
    clean_set = set()
    lot_size = len(noisy_targets) / k

    # FIXME: might not be taking the last index, so just have a look 
    # for i in range(k):
    #     print("Batch No. "+  str(i+1) + " Starting")
    #     tempArr, _ = fine(current_features=train_data[lot_size*i:lot_size*(i+1)], current_labels=noisy_targets[lot_size*i:lot_size*(i+1)], fit="gmm")
    #     tempArr = [(val + lot_size*i) for val in tempArr]
    #     clean_set |= set(tempArr)

    clean_idxs = fine(train_data, noisy_targets, "gmm")

    clean_idxs = list(clean_set)
    noisy_idxs = []

    for idx in range(0, len(noisy_targets)):
        if idx not in clean_set:
            noisy_idxs.append(idx)

    return clean_idxs, noisy_idxs

# FIXME: Pass enum type in future 
# FIXME: Try to come up with something similar afterwards
def return_confident_indexes(model, train_data, clean_targets, noisy_targets, isFine=False, k=100):

    if isFine:
        return helperFunctionForFINE(train_data, noisy_targets, k)

    else:

        predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
        predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        softouts = predict_softmax(predict_loader, model)
        return splite_confident(softouts, clean_targets, noisy_targets)

    return None, None
		


# takes the model, train data, clean targets, and noisy targets to return labeled, unlabeles loaders with class weights 

def update_trainloader(model, train_data, clean_targets, noisy_targets, isFine=False):

	print("Update Train Loader is called")
	
	confident_indexs, unconfident_indexs = return_confident_indexes(model, train_data, clean_targets, noisy_targets, isFine)

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
	print("Category", train_nums, "precent", class_weights)
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
train_dataset = Train_Dataset(data, noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)

model = create_model(num_classes=args.num_class)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
if args.optim == 'cos':
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs, args.lr / 100)
else:
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

best_test_acc = 0

# TODO: remove this after testing 

args.T1 = 10
args.T2 = 5
args.num_epochs = 15

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score

def trial_shiv(train_data, noisy_labels, clean_labels):

    print("Before: ", train_data.shape)

    reshaped_train=train_data.reshape(50000,3072)

    print("After: ", reshaped_train.shape)

    n_categories= len(np.unique(clean_labels))
    pca = PCA(n_components=10)
    # kmeans = KMeans(n_clusters=n_categories,max_iter=200)
    gmm = GaussianMixture(n_components=n_categories, covariance_type='diag', tol=1e-6, max_iter=10)

    predictor = Pipeline([('pca', pca), ('gmm', gmm)])
    predict = predictor.fit(reshaped_train).predict(reshaped_train)

    print(f1_score(clean_labels, predict, average="macro"))

    return

# trial_shiv(data, noisy_labels, clean_labels)

# TODO: you can print this afterwards
# print("Some of the parameters: ", args.T1, args.T2, args.num_epochs)

def evaluate_accuracy(model, train_data, clean_targets, noisy_targets, k=100):

	print("Evaluate Accuracy function is called")

	confident_idxs_FINE, unconfident_idxs_FINE = return_confident_indexes(model, train_data, clean_targets, noisy_targets, True, k)
	
	print("PES with FINE: ", len(confident_idxs_FINE), len(unconfident_idxs_FINE))

	print(confident_idxs_FINE)

	return 

# FIXME: for testing only passing on 10 labels 
K = 2 # batch size
evaluate_accuracy(model, data[:2], clean_labels[:2], noisy_labels[:2], K)

quit()

for epoch in range(args.num_epochs):

	print("Epoch: ", epoch)
    # training per say
	if epoch < args.T1:
		train(model, train_loader, optimizer, ceriation, epoch)
	else:
		if epoch == args.T1:
			model = noisy_refine(model, train_loader, 0, args.T2)
			while (K <= 5000):
				evaluate_accuracy(model, data, clean_labels, noisy_labels, K)
				K *= 10

        # arguments required for mix match that update trainloader returns
		labeled_trainloader, unlabeled_trainloader, class_weights = update_trainloader(model, data, clean_labels, noisy_labels, True)

        # mixmatch to learn from the clean models and make the noisy models correct 
		MixMatch_train(epoch, model, optimizer, labeled_trainloader, unlabeled_trainloader, class_weights)

    # validation 
	_, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
	best_test_acc = test_acc if best_test_acc < test_acc else best_test_acc
	scheduler.step()

print(getTime(), "Best Test Acc:", best_test_acc)
