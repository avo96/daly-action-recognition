import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Detector(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Detector, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, num_classes))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def train_epocs(model, optimizer, train_data, test_data, epochs=10,C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y, box in train_data:
            batch = y.shape[0]
            x = x.cuda().float()
            y = y.cuda()
            box = box.cuda().float()
            out_class, out_box = model(x)
            loss_class = F.cross_entropy(out_class, y, reduction="sum")
            loss_bb = F.l1_loss(out_box, box, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        val_loss, val_acc = test_metrics(model, test_data, C)
        torch.save(model.state_dict(),'weights.pth')
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
    return sum_loss/total

def test_metrics(model, test_data, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y, box in test_data:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda()
        box = box.cuda().float()
        out_class, out_box = model(x)
        loss_class = F.cross_entropy(out_class, y, reduction="sum")
        loss_bb = F.l1_loss(out_box, box, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total