import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions.bernoulli
import warnings
warnings.filterwarnings('ignore')


class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()

        self.fc1 = nn.Linear(784, 400)

        self.fc2 = nn.Linear(400, 400)

        self.fc3 = nn.Linear(400, 10)


    def forward(self, x, T=1):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = F.relu(h)

        h = self.fc2(h)
        h = F.relu(h)

        h = self.fc3(h)
        out, log_out = F.softmax(h/T), F.log_softmax(h/T)
        return out, log_out

class StudentModel(nn.Module):
    def __init__(self, student_hidden_size, student_hidden_size_2, student_dropout_rate, return_entropy=False):
        super(StudentModel, self).__init__()
        self.student_dropout_rate = student_dropout_rate
        self.return_entropy = return_entropy
        self.fc1 = nn.Linear(28*28, student_hidden_size)
        self.fc2 = nn.Linear(student_hidden_size, student_hidden_size_2)
        self.fc3 = nn.Linear(student_hidden_size_2, 10)
        self.fc4 = nn.Linear(student_hidden_size_2, 1)

    def forward(self, x, T=1, return_entropy=False):
        x = x.view(-1, 28*28)
        x = F.dropout(F.relu(self.fc1(x)), )
        x = F.relu(self.fc2(x))
        entropy_out = self.fc3(x)
        x = self.fc3(x)
        out, log_out = F.softmax(x/T), F.log_softmax(x/T)
        if return_entropy or self.return_entropy:
            return out, log_out, entropy_out
        else:
            return out, log_out


class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 4, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 4, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20 * 4 * 4, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x, T=1):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x/T), F.log_softmax(x/T)

class StudentCNN(nn.Module):
    def __init__(self, student_multiplication_factor_1, student_multiplication_factor_2, student_dropout_rate, return_entropy=False):
        super(StudentCNN, self).__init__()
        self.dropout_rate = student_dropout_rate
        self.return_entropy = return_entropy
        self.conv1 = nn.Conv2d(1, int(10 * student_multiplication_factor_1), 4, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(int(10 * student_multiplication_factor_1), int(20 * student_multiplication_factor_1), 4, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(int(20 * 4 * 4 * student_multiplication_factor_1), int(80 * student_multiplication_factor_2))
        self.fc2 = nn.Linear(int(80 * student_multiplication_factor_2), 10)
        self.fc3 = nn.Linear(int(80 * student_multiplication_factor_2), 1)
        # self.fc4 = nn.Linear(int(80 * student_multiplication_factor_2), 20)

    def forward(self, x, T=1, return_entropy=False):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.dropout2d(self.pool1(x), p=self.student_dropout_rate, training=self.training)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(self.pool2(x), p=self.student_dropout_rate, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=self.student_dropout_rate, training=self.training)
        entropy_out = self.fc3(x)
        # percentile_proba_out = F.sigmoid(self.fc4(x))
        x = self.fc2(x)
        if return_entropy or self.return_entropy:
            return F.softmax(x/T), F.log_softmax(x/T), entropy_out
        else:
            return F.softmax(x/T), F.log_softmax(x/T)


class TeacherCNNCIFAR(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1   = nn.Linear(32*5*5, 200)
        self.fc2   = nn.Linear(200, 50)
        self.fc3   = nn.Linear(50, 10)
        # self.fc4   = nn.Linear(84, 10)
    def forward(self, x, T=1):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.softmax(out/T), F.log_softmax(out/T)


class StudentCNNCIFAR(nn.Module):
    def __init__(self, student_multiplication_factor_1, student_multiplication_factor_2, student_dropout_rate, return_entropy=False):
        super(StudentCNN, self).__init__()
        self.dropout_rate = student_dropout_rate
        self.return_entropy = return_entropy
        self.conv1 = nn.Conv2d(3, int(16*student_multiplication_factor_1), 5)
        # self.bn1 = nn.BatchNorm2d(int(32 * student_multiplication_factor_1))
        self.conv2 = nn.Conv2d(int(16*student_multiplication_factor_1), int(32*student_multiplication_factor_1), 5)
        # self.bn2 = nn.BatchNorm2d(int(64*student_multiplication_factor_1))
        self.fc1   = nn.Linear(int(32*student_multiplication_factor_1)*5*5, int(200 * student_multiplication_factor_2))
        # self.bn3 = nn.BatchNorm1d(int(800 * student_multiplication_factor_2))
        self.fc2   = nn.Linear(int(200 * student_multiplication_factor_2), int(50 * student_multiplication_factor_2))
        # self.bn4 = nn.BatchNorm1d(int(120 * student_multiplication_factor_2))
        self.fc3   = nn.Linear(int(50 * student_multiplication_factor_2), 10)
        # self.bn5 = nn.BatchNorm1d(int(84 * student_multiplication_factor_2))
        self.fc4   = nn.Linear(int(50 * student_multiplication_factor_2), 1)

    def forward(self, x, T=1, return_entropy=False):
        out = F.relu(self.conv1(x))
        out = F.dropout2d(F.max_pool2d(out, 2), p=self.student_dropout_rate, training=self.training)
        out = F.relu(self.conv2(out))
        out = F.dropout2d(F.max_pool2d(out, 2), p=self.student_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.fc1(out)), p=self.student_dropout_rate, training=self.training)
        out = F.dropout(F.relu(self.fc2(out)), p=self.student_dropout_rate, training=self.training)
        entropy_out = self.fc4(out)
        out = self.fc3(out)
        if return_entropy or self.return_entropy:
            return F.softmax(out), F.log_softmax(out), entropy_out
        else:
            return F.softmax(out), F.log_softmax(out)