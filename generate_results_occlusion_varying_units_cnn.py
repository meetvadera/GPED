import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.utils.data as utils
import torch.distributions.bernoulli
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from collections import OrderedDict
from matplotlib import pyplot as plt
import advertorch.attacks as attacks
import seaborn as sns
import argparse
import csv
from tdigest import TDigest
import time
from quantile import Estimator
from models import TeacherCNN, StudentCNN
import warnings
warnings.filterwarnings('ignore')

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--teacher_lr', type=float)
parser.add_argument('--teacher_wd', type=float)
parser.add_argument('--student_lr', type=float)
parser.add_argument('--student_wd', type=float)
parser.add_argument('--student_hidden_size', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--burn_in_time', type=float)
parser.add_argument('--num_student_paths', type=int)
parser.add_argument('--num_training_samples', type=int)
parser.add_argument('--mask_size', type=int)
parser.add_argument('--student_multiplication_factor_1', type=float)
parser.add_argument('--student_multiplication_factor_2', type=float)
parser.add_argument('--student_dropout_rate', type=float)
parser.add_argument('--seed', type=int)
args = parser.parse_args()
torch.manual_seed(args.seed)

train_data, train_labels = np.load('../data/mnist/mnist_{}k_train_data.npy'.format(args.num_training_samples)), np.load('../data/mnist/mnist_{}k_train_labels.npy'.format(args.num_training_samples))
student_train_data = np.load('../data/mnist/mnist_60k_train_data.npy')
test_data, test_labels = np.load('../data/mnist/mnist_test_data.npy'), np.load('../data/mnist/mnist_test_labels.npy')

patch_size = args.mask_size
for i, image_array in enumerate(train_data):
    image_size = image_array.shape[1]
    mask_start_x = np.random.randint(0, image_size - patch_size)
    mask_start_y = np.random.randint(0, image_size - patch_size)
    train_data[i, mask_start_x:mask_start_x + patch_size, mask_start_y:mask_start_y + patch_size]  *= 0

for i, image_array in enumerate(student_train_data):
    image_size = image_array.shape[1]
    mask_start_x = np.random.randint(0, image_size - patch_size)
    mask_start_y = np.random.randint(0, image_size - patch_size)
    student_train_data[i, mask_start_x:mask_start_x + patch_size, mask_start_y:mask_start_y + patch_size]  *= 0

for i, image_array in enumerate(test_data):
    image_size = image_array.shape[1]
    mask_start_x = np.random.randint(0, image_size - patch_size)
    mask_start_y = np.random.randint(0, image_size - patch_size)
    test_data[i, mask_start_x:mask_start_x + patch_size, mask_start_y:mask_start_y + patch_size]  *= 0

train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels.squeeze()).long()
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels.squeeze()).long()
student_train_data = torch.from_numpy(student_train_data).float()
student_train_data = student_train_data.cuda()

train_dataset = utils.TensorDataset(train_data, train_labels)
test_dataset = utils.TensorDataset(test_data, test_labels)
minibatch_size = 100
dataloader_args = dict(shuffle=True, batch_size=minibatch_size,num_workers=8, pin_memory=True)
train_loader = dataloader.DataLoader(train_dataset, **dataloader_args)
test_loader = dataloader.DataLoader(test_dataset, shuffle=True, batch_size=10000, num_workers=8, pin_memory=True)



teacher_model.train()
t = 0.
teacher_model = TeacherCNN()
student_model = StudentCNN(args.student_multiplication_factor_1, args.student_multiplication_factor_2, args.student_dropout_rate)
print(student_model)
teacher_model.cuda()
student_model.cuda()
losses = []
teacher_lr = args.teacher_lr
student_lr = args.student_lr
teacher_wd = args.teacher_wd
student_wd = args.student_wd
perturb_deviation = args.perturb_deviation
student_optimizer = optim.Adam(student_model.parameters(), lr=student_lr, weight_decay=student_wd)
#student_optimizer = optim.SGD(student_model.parameters(), lr=student_lr, weight_decay=student_wd, momentum=0.9, nesterov=True)
##num_epochs = args.num_epochs
num_epochs = args.num_epochs
burn_in_time = args.burn_in_time
test_set, test_labels = test_data, test_labels
test_set = test_set.cuda()
test_proba = torch.zeros((test_set.shape[0], 10))
student_optimizer_scheduler = optim.lr_scheduler.StepLR(student_optimizer, step_size=100, gamma=0.5)
train_proba = torch.zeros(student_train_data.shape[0], 10)
entropy_mean_train = torch.zeros(student_train_data.shape[0], 1)
t_dicts_train = []
timing_iterations = 0
total_time = 0.
for i in range(len(student_train_data)):
    t_dicts_train.append([Estimator(*[(0.25, 0.1), (0.75, 0.1)]) for t in range(10)])
t_dicts_test = []
for i in range(len(test_data)):
    t_dicts_test.append([Estimator(*[(0.25, 0.1), (0.75, 0.1)]) for t in range(10)])
student_train_data_idx_count = torch.zeros(student_train_data.shape[0], 1)
entropy_mean_test = torch.zeros(test_data.shape[0], 1)
num_epochs = int((1e6)//(args.num_training_samples * 1000/100))
attack = attacks.PGDAttack(student_model, loss_fn=nn.NLLLoss(), eps=args.attack_eps, eps_iter=0.005)
for epoch in range(num_epochs):
    student_model.train()
    student_optimizer_scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data, target = Variable(data.cuda()), Variable(target.cuda())
        student_optimizer.zero_grad()
        _, teacher_log_prediction = teacher_model(data)
        teacher_loss = F.nll_loss(teacher_log_prediction, target)
        for w in teacher_model.parameters():
            if w.grad is not None:
                w.grad.data.zero_()
        teacher_loss.backward(retain_graph=True)
        for w in teacher_model.parameters():
            w.data += -teacher_lr/2. * (w.grad.data * len(train_dataset) + teacher_wd * w.data) + torch.normal(torch.zeros_like(w.data), torch.ones_like(w.data) * math.sqrt(teacher_lr)).cuda()
        if t >= burn_in_time:
            test_proba_iteration, _ = teacher_model(test_set)
            # psi_array.append(float(F.nll_loss(test_proba_iteration, test_labels_aux).cpu().detach()))
            test_proba += test_proba_iteration.cpu().detach()
            entropy_mean_test += -torch.sum((test_proba_iteration + 1e-7) * torch.log((test_proba_iteration + 1e-7)), dim=1, keepdim=True).cpu().detach()
            if t % 100 == 0:
                test_proba_iteration_numpy = test_proba_iteration.clone().detach().cpu().numpy()
                # for k in range(len(test_proba_iteration_numpy)):
                #     for l in range(10):
                #         t_dicts_test[k][l].observe(test_proba_iteration_numpy[k, l])
                # student_data = data + torch.normal(torch.zeros_like(data), torch.ones_like(data) * perturb_deviation).cuda()
                student_minibatch_idxs = torch.randint(0, 60000, (minibatch_size,))
                student_data = Variable(student_train_data[student_minibatch_idxs]) + torch.normal(torch.zeros_like(data), torch.ones_like(data) * perturb_deviation).cuda()
                student_data.clamp_(0, 1)
                _, teacher_log_prediction = teacher_model(student_data)
                teacher_prediction = torch.exp(teacher_log_prediction.detach())
                teacher_prediction_numpy = teacher_prediction.clone().detach().cpu().numpy()
                # teacher_median_prediction_targets = torch.zeros(student_data.shape[0], 20).cuda()
                # teacher_iq_targets = torch.zeros(student_data.shape[0], 20).cuda()
                # for k in range(len(teacher_prediction_numpy)):
                #     for l in range(10):
                #         t_dicts_train[student_minibatch_idxs[k]][l].observe(teacher_prediction_numpy[k, l])
                #         teacher_median_prediction_targets[k, 2*l] = float(t_dicts_train[student_minibatch_idxs[k]][l].query(0.25))
                #         teacher_median_prediction_targets[k, 2*l+1] = float(t_dicts_train[student_minibatch_idxs[k]][l].query(0.75))
                #         teacher_iq_targets[k, [2*l, 2*l+1]] = float(t_dicts_train[student_minibatch_idxs[k]][l].query(0.75) - t_dicts_train[student_minibatch_idxs[k]][l].query(0.25))
                teacher_entropy_current_iteration = -torch.sum((teacher_prediction + 1e-7) * torch.log(teacher_prediction + 1e-7), dim=1, keepdim=True)
                entropy_mean_train[student_minibatch_idxs] += teacher_entropy_current_iteration.cpu().detach()
                train_proba[student_minibatch_idxs] += teacher_prediction.cpu().detach()
                student_train_data_idx_count[student_minibatch_idxs] += 1
                teacher_entropy_target = entropy_mean_train[student_minibatch_idxs]/student_train_data_idx_count[student_minibatch_idxs]
                train_proba_target = train_proba[student_minibatch_idxs]/student_train_data_idx_count[student_minibatch_idxs]
                _, student_log_prediction, student_log_entropy = student_model(student_data, return_entropy=True)
                student_entropy = torch.exp(student_log_entropy)
                # student_loss = -torch.mean(torch.sum(teacher_prediction * student_log_prediction, dim=1))
                student_loss = -torch.mean(torch.sum(train_proba_target.cuda() * student_log_prediction, dim=1))
 #               student_loss = F.nll_loss(student_log_prediction, train_labels[student_minibatch_idxs].cuda())
#                entropy_loss = torch.mean(torch.abs(teacher_entropy_target.cuda() - student_entropy))
                entropy_loss = torch.mean(torch.abs(teacher_entropy_current_iteration - student_entropy))
                # total_loss = entropy_loss + student_loss 
                total_loss = student_loss
                total_loss.backward(retain_graph=True)
                student_optimizer.step()
#                torch.save(teacher_model.state_dict(), 'teacher_models/teacher_model_state_dict_iter_%d.pt' % int(t))

        t += 1
        # Display
        if batch_idx % (args.num_training_samples * 10) == 0 and t>=burn_in_time:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tTeacher Loss: {:.6f}\tStudent Loss Classification: {:.6f}\tStudent Loss Entropy: {:.6f}\n'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                teacher_loss.data,
                student_loss.data,
                entropy_loss.data),
                end='')
    total_correct = 0
    avg_loss = 0.0
#     for i, (images, labels) in enumerate(test_loader):
    if t > burn_in_time and (epoch % 100 == 0 or epoch == num_epochs - 1):
        student_model.eval()
        _, log_output, student_log_entropy_test = student_model(test_set, return_entropy=True)
        # adv_examples_test = attack.fgsm(test_set, test_labels.cuda(), args.attack_eps, 0, 1)
#        adv_examples_test = attack.perturb(test_set, test_labels.cuda())
#        _, adv_log_output, _ , _= student_model(adv_examples_test)
        output = torch.exp(log_output)
#        adv_loss = F.nll_loss(adv_log_output.cpu(), test_labels)
#        adv_pred = adv_log_output.detach().cpu().max(1)[1]
        avg_loss += F.nll_loss(log_output.cpu(), test_labels, reduction='sum')
        pred = output.detach().max(1)[1].cpu()
        total_correct += pred.eq(test_labels.view_as(pred)).sum()
#        adv_total_correct = adv_pred.eq(test_labels.view_as(pred)).sum()
        entropy_diff_teacher_student_test = torch.mean(torch.abs(torch.exp(student_log_entropy_test).cpu().detach() - entropy_mean_test/(t-burn_in_time)))
        # teacher_median_predictions_test = torch.zeros(len(t_dicts_test), 20)
        # teacher_iq_test = torch.zeros(len(t_dicts_test), 20)
        # for k in range(len(t_dicts_test)):
        #     for l in range(10):
        #         teacher_median_predictions_test[k, 2*l] = float(t_dicts_test[k][l].query(0.25))
        #         teacher_median_predictions_test[k, 2*l+1] = float(t_dicts_test[k][l].query(0.75))
        # median_prediction_loss = torch.mean(torch.sum(torch.abs(teacher_median_predictions_test - student_median_proba_value_test.cpu().detach()), dim=1))
        avg_loss /= len(test_labels)
        teacher_loss = F.nll_loss(torch.log(test_proba/(t - burn_in_time)), test_labels)
        teacher_predictions = (test_proba/(t - burn_in_time)).max(1)[1]
        teacher_accuracy = float(teacher_predictions.eq(test_labels).sum())/len(test_labels)
        student_teacher_ce_loss = -torch.mean(torch.sum(test_proba/(t - burn_in_time) * log_output.cpu(), dim=1))
        kl_div_test = student_teacher_ce_loss + torch.mean(torch.sum(test_proba/(t - burn_in_time) * torch.log(test_proba/(t - burn_in_time)), dim=1))
        inverse_kl_loss = -torch.mean(torch.sum(torch.log(test_proba/(t - burn_in_time)) * output.cpu(), dim=1)) + torch.mean(torch.sum(log_output.cpu() * output.cpu(), dim=1))
        mae_loss = torch.mean(torch.abs(test_proba/(t - burn_in_time) - output.cpu()))
        print('Test Avg. Loss using student: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(test_labels)))
        print('Test Avg. Loss using teacher: %f, Accuracy: %f' % (teacher_loss.detach().cpu().item(), teacher_accuracy))
        print('Avg. CE Loss between teacher and student: %f' % (student_teacher_ce_loss.detach().cpu().item()))
        print('Inverse KL Loss between teacher and student: %f' % (inverse_kl_loss.detach().cpu().item()))
        print('MAE  Loss between teacher and student: %f' % (mae_loss.detach().cpu().item()))
        print('Test Diff in mean entropies: %f' % entropy_diff_teacher_student_test.item())
#        print('Adversarial loss on student: %f, Accuracy: %f' % (adv_loss.detach().item(), float(adv_total_correct)/len(test_labels)))
        # print('Test Diff in medians of class: %f' % median_prediction_loss.item())
# psi_array = np.array(psi_array)
# if args.file_prefix is None:
#     #np.save('psi_array_single_chain_{}k_samples_{}_mask.npy'.format(args.num_training_samples, patch_size), psi_array)
#     np.save('teacher_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.num_training_samples, patch_size), (test_proba/(t - burn_in_time)).cpu().numpy())
#     np.save('teacher_train_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.num_training_samples, patch_size), (train_proba/(t - burn_in_time)).cpu().numpy())
#     np.save('student_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.num_training_samples, patch_size), output.cpu().detach().numpy())
# else:
#     #np.save('{}_psi_array_single_chain_{}k_samples_{}_mask.npy'.format(args.num_training_samples, patch_size), psi_array)
#     np.save('different_loss_results/{}_teacher_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.file_prefix, args.num_training_samples, patch_size), (test_proba/(t - burn_in_time)).cpu().numpy())
#     np.save('different_loss_results/{}_teacher_train_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.file_prefix, args.num_training_samples, patch_size), (train_proba/ (t - burn_in_time)).cpu().numpy())
#     np.save('different_loss_results/{}_student_predictions_single_chain_{}k_samples_{}_mask.npy'.format(args.file_prefix, args.num_training_samples, patch_size), output.cpu().detach().numpy())
# kl_div_test_array = -torch.sum(test_proba/(t - burn_in_time) * log_output.cpu(), dim=1) + torch.sum(test_proba/(t - burn_in_time) * torch.log(test_proba/(t - burn_in_time)), dim=1)
# kl_div_test_array = kl_div_test_array.detach().cpu().numpy()
# np.save('iclr_results/mnist_60k_0_mask_kl_div_array.npy', nll_test_array)
# nll_test_array = F.nll_loss(log_output.cpu(), test_labels, reduction='none') - F.nll_loss(torch.log(test_proba/(t - burn_in_time)), test_labels, reduction='none')
# nll_test_array = nll_test_array.detach().cpu().numpy()
# np.save('iclr_results/mnist_60k_0_mask_nll_diff_array.npy', nll_test_array)
# entropy_diff_teacher_student_test_array = torch.abs(torch.exp(student_log_entropy_test).cpu().detach() - entropy_mean_test/(t-burn_in_time))
# np.save('iclr_results/mnist_60k_0_mask_entropy_diff_array.npy', entropy_diff_teacher_student_test_array)
#torch.save(student_model.state_dict(), 'student_models/student_model_cnn_mnist_for_quantization_entropy.pt')
with open('cnn_mnist_reduced_variance_estimator_results_predictive_means.csv', 'a+') as csvFile:
    writer =  csv.writer(csvFile, dialect='excel')
    writer.writerow(['kl', args.num_training_samples, args.mask_size, args.attack_eps, args.perturb_deviation, args.student_multiplication_factor_1, args.student_multiplication_factor_2, teacher_loss.detach().cpu().item(), teacher_accuracy, avg_loss.detach().cpu().item(), float(total_correct) / len(test_labels), kl_div_test.detach().cpu().item(), mae_loss.detach().cpu().item(), entropy_diff_teacher_student_test.detach().cpu().item(), 0])
