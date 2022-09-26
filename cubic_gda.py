import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from math import sqrt
from net import Net
from device import DeviceDataLoader


def cubic_gda(gamma=1.0, batch_size=128, num_epochs=10, num_ascent_epochs=20, lr_1=0.1, lr_2=0.01, lr_3=0.01,
                lr_4=0.01):
    ## lr_1 is used for ascent, lr_2 is used for the cubic step, lr_3 and lr_4 are used for the gda step to solve the cubic problem
    ## here we set B1=B2=B11=B12=B21=B22=batch_size, and in each epoch we only solve one minibatch

    # Step 0. Initialization

    transform = torchvision.transforms.Compose([
        # transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
    test_loader = DeviceDataLoader(test_loader, device)

    net = Net()
    torch.save(net.state_dict(), "tmp.pth")
    net.to(device)

    train_loss = []
    acc = []
    robust_acc = []

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.float().to(device)
            labels.float().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc.append(correct / total)

    for i in range(num_epochs):
        print("The %d-th epoch starts." % i)
        start_time = time.time()

        ## step 1, num_ascent_epochs steps of gradient ascend on kxi

        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_loader = DeviceDataLoader(train_loader, device)

        net = Net()
        net.load_state_dict(torch.load("tmp.pth"))
        net.to(device)

        for images, labels in train_loader:
            kxi = torch.clone(images)  ## first set kxi=x
            x = torch.clone(images)
            targets = labels
            break

        kxi.requires_grad = True
        x.requires_grad = False
        targets.requires_grad = False
        for param in net.parameters():
            param.requires_grad = False

        for j in range(num_ascent_epochs):
            output = net(kxi)
            loss1 = F.cross_entropy(output, targets)
            loss2 = torch.sum((kxi - x) ** 2) / batch_size
            loss = loss1 - gamma * loss2
            loss.backward()
            kxi.data = kxi.data + lr_1 * kxi.grad.data
            kxi.grad.data.zero_()

        ## step 2, use cubic Regularization to calculate the new parameters of the neural network

        kxi.requires_grad = True
        for param in net.parameters():
            param.requires_grad = True
        output = net(kxi)
        loss1 = F.cross_entropy(output, targets)
        loss2 = torch.sum((x - kxi) ** 2) / batch_size
        loss = loss1 - gamma * loss2

        f_xgrad = torch.autograd.grad(loss,
                                      net.parameters())  ## use to store grad_x f(x,y), here x means the net parameters
        net.zero_grad()

        output = net(kxi)
        loss1 = F.cross_entropy(output, targets)
        loss2 = torch.sum((x - kxi) ** 2) / batch_size
        loss = loss1 - gamma * loss2

        fx_grad = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        fy_grad = torch.autograd.grad(loss, kxi, create_graph=True)

        ## initialze s and v

        # s has the same shape with net parameters, v has the same shape with kxi
        # in first epoch set s and v to zero, then in after epochs we set the initialize value of s and v to the result of last epoch
        if i == 0:
            s = []
            f_xgrad = list(f_xgrad)
            for i in range(len(f_xgrad)):
                s.append(torch.zeros(f_xgrad[i].shape, device='cuda', requires_grad=True))
            v = torch.zeros(kxi.shape, device='cuda', requires_grad=True)
        else:
            s = s_copy.copy()
            f_xgrad = list(f_xgrad)
            v = torch.clone(v_copy)

        ## use gda to solve the cubic problem

        ## Cubic-Subsolver via Gradient Descent(using algorithm from Stochastic Cubic Regularization for Fast Nonconvex Optimization)

        ## in case the cubic methed do not converge, we seperate the algorithm into two conditions

        f_xgrad_norm_2 = torch.Tensor([(f_xgrad[i] ** 2).sum() for i in range(len(f_xgrad))]).sum()

        if f_xgrad_norm_2 > 20:

            print("f_xgrad is too big! Use alternative method.")

            for i in range(500):
                y_grad_v = (fy_grad[0] * v).sum()
                yy_v = torch.autograd.grad(y_grad_v, kxi, create_graph=True)
                zeta1 = 0.5 * (yy_v[0] * v).sum()

                xy_v = torch.autograd.grad(y_grad_v, net.parameters(), create_graph=True)
                xy_v = list(xy_v)
                zeta2 = 0
                for i in range(len(s)):
                    zeta2 += (f_xgrad[i] * xy_v[i]).sum()
                x_grad_x_grad = 0
                for i in range(len(s)):
                    x_grad_x_grad += (f_xgrad[i] * fx_grad[i]).sum()
                xx_x_grad = torch.autograd.grad(x_grad_x_grad, net.parameters(), create_graph=True)
                zeta3 = 0
                for i in range(len(s)):
                    zeta3 += 0.5 * (xx_x_grad[i] * f_xgrad[i]).sum()
                zeta = zeta1 + zeta2 + zeta3

                v_grad = torch.autograd.grad(zeta, v, create_graph=False, allow_unused=True)

                if (v_grad[0] ** 2).sum() < 0.00001:
                    break

                v.data = v.data + 1 * v_grad[0]

            Rc = -zeta * lr_2 / (f_xgrad_norm_2) + sqrt(
                (zeta * lr_2 / (f_xgrad_norm_2)) ** 2 + 2 * lr_2 * sqrt(f_xgrad_norm_2))
            for i in range(len(s)):
                s[i] = -Rc * f_xgrad[i] / sqrt(f_xgrad_norm_2)

            s_copy = s.copy()
            v_copy = torch.clone(v)

        else:
            ## use classical gda to solve cubic method
            for i in range(500):
                for _ in range(5):
                    cubic1 = 0
                    for i in range(len(s)):
                        cubic1 += (s[i] * f_xgrad[i]).sum()

                    y_grad_v = (fy_grad[0] * v).sum()
                    yy_v = torch.autograd.grad(y_grad_v, kxi, create_graph=True)
                    zeta1 = 0.5 * (yy_v[0] * v).sum()
                    xy_v = torch.autograd.grad(y_grad_v, net.parameters(), create_graph=True)
                    xy_v = list(xy_v)
                    zeta2 = 0
                    for i in range(len(s)):
                        zeta2 += (s[i] * xy_v[i]).sum()
                    x_grad_s = 0
                    for i in range(len(s)):
                        x_grad_s += (s[i] * fx_grad[i]).sum()
                    xx_s = torch.autograd.grad(x_grad_s, net.parameters(), create_graph=True)
                    zeta3 = 0
                    for i in range(len(s)):
                        zeta3 += 0.5 * (xx_s[i] * s[i]).sum()
                    zeta = zeta1 + zeta2 + zeta3

                    cubic3 = 0
                    for i in range(len(s)):
                        cubic3 += (s[i] * s[i]).sum()
                    cubic3 = cubic3 ** (3 / 2) / (6 * lr_2)

                    cubic = cubic1 + zeta + cubic3

                    v_grad = torch.autograd.grad(cubic, v, create_graph=False, allow_unused=True)
                    v.data = v.data + lr_3 * v_grad[0]

                cubic1 = 0
                for i in range(len(s)):
                    cubic1 += (s[i] * f_xgrad[i]).sum()

                y_grad_v = (fy_grad[0] * v).sum()
                yy_v = torch.autograd.grad(y_grad_v, kxi, create_graph=True)
                zeta1 = 0.5 * (yy_v[0] * v).sum()
                xy_v = torch.autograd.grad(y_grad_v, net.parameters(), create_graph=True)
                xy_v = list(xy_v)
                zeta2 = 0
                for i in range(len(s)):
                    zeta2 += (s[i] * xy_v[i]).sum()
                x_grad_s = 0
                for i in range(len(s)):
                    x_grad_s += (s[i] * fx_grad[i]).sum()
                xx_s = torch.autograd.grad(x_grad_s, net.parameters(), create_graph=True)
                zeta3 = 0
                for i in range(len(s)):
                    zeta3 += 0.5 * (xx_s[i] * s[i]).sum()
                zeta = zeta1 + zeta2 + zeta3

                cubic3 = 0
                for i in range(len(s)):
                    cubic3 += (s[i] * s[i]).sum()
                cubic3 = cubic3 ** (3 / 2) / (6 * lr_2)

                cubic = cubic1 + zeta + cubic3

                s_grad = torch.autograd.grad(cubic, s, retain_graph=True, create_graph=False, allow_unused=True)
                v_grad = torch.autograd.grad(cubic, v, create_graph=False, allow_unused=True)
                s_grad = list(s_grad)

                s_grad_norm = 0
                for i in range(len(s)):
                    s_grad_norm += (s_grad[i] * s_grad[i]).sum()
                v_grad_norm = (v_grad[0] * v_grad[0]).sum()

                if s_grad_norm < 0.0001 and v_grad_norm < 0.0001:
                    break

                for i in range(len(s)):
                    s[i].data = s[i].data - lr_4 * s_grad[i]

            s_copy = s.copy()
            v_copy = torch.clone(v)

        tmp_net = Net()
        tmp_net.load_state_dict(torch.load("tmp.pth"))
        tmp_net.to(device)
        tmp_list = []
        num_layer = 0
        for param in tmp_net.parameters():
            tmp_list.append(torch.tensor(param) + s[num_layer])
            num_layer += 1

        num_layer = 0
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(tmp_list[num_layer])
                num_layer += 1

        torch.save(net.state_dict(), "tmp.pth")

        ## step 3 variation step

        ## compute the phi(x)
        # use 100-step gradient ascent as estimate loss
        est_net = Net()
        est_net.load_state_dict(torch.load("tmp.pth"))
        est_net.to(device)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=0)
        test_loader = DeviceDataLoader(test_loader, device)

        est_loss = 0.0
        for param in est_net.parameters():
            param.requires_grad = False
        for data in test_loader:
            images, labels = data
            kxi = torch.clone(images)
            kxi.requires_grad = True
            for j in range(100):
                output = est_net(kxi)
                loss1 = F.cross_entropy(output, labels)
                loss2 = torch.sum((kxi - images) ** 2) / 2000
                loss = loss1 - gamma * loss2
                loss.backward()
                kxi.data = kxi.data + 0.1 * kxi.grad.data
                kxi.grad.data.zero_()
            break
        est_loss = loss.item()
        train_loss.append(est_loss)

        ## compute accuracy
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

        test_net = Net()
        test_net.load_state_dict(torch.load("tmp.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.float()
                labels.float()
                outputs = test_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc.append(correct / total)

        ## test robust accuracy

        test_loader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=0)
        test_loader = DeviceDataLoader(test_loader, device)

        test_net = Net()
        test_net.to(device)
        test_net.load_state_dict(torch.load("tmp.pth"))

        correct_robust = 0
        total_robust = 0
        for data in test_loader:
            images, labels = data
            kxi = torch.clone(images)
            kxi.requires_grad = True
            for j in range(10):
                output = test_net(kxi)
                loss1 = F.cross_entropy(output, labels)
                loss2 = torch.sum((kxi - images) ** 2) / 2000
                loss = loss1 - gamma * loss2
                loss.backward()
                kxi.data = kxi.data + lr_1 * kxi.grad.data
                kxi.grad.data.zero_()

            with torch.no_grad():
                outputs = test_net(kxi)
                _, predicted = torch.max(outputs.data, 1)
                total_robust += labels.size(0)
                correct_robust += (predicted == labels).sum().item()
            break
        robust_acc.append(correct_robust / total_robust)

        print("train loss:", est_loss, "accuracy:", correct / total, "robust accuracy ", correct_robust / total_robust)
        print("Epoch ends. Time spent:", time.time() - start_time)

    return train_loss, acc, robust_acc