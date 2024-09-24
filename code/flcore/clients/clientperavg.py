import numpy as np
import pandas as pd
import torch
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
from ..trainmodel.models import BinaryLogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.fairness_loss import IndividualFairnessLoss, GroupFairnessLoss, HybridFairnessLoss


class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self, new_local_epochs):
        trainloader = self.load_train_data(self.batch_size * 2)
        start_time = time.time()

        self.model.train()
        max_local_epochs = new_local_epochs[self.id]
        if self.train_slow:
            max_local_epochs = np.random.randint(1, new_local_epochs // 2)
        for step in range(max_local_epochs):  # local update
            for X, Y, sensitive_feature in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))
                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    y = y.to(torch.float)
                if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                        or isinstance(self.loss, HybridFairnessLoss)):
                    df = pd.DataFrame(x.numpy())
                    df[self.sensitive_feature] = sensitive_feature[:self.batch_size].to(self.device)
                    df['y'] = y.numpy()
                    loss = self.loss(output, y, df, self.sensitive_feature, self.model)
                else:
                    loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    y = y.to(torch.float)
                if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                        or isinstance(self.loss, HybridFairnessLoss)):
                    df = pd.DataFrame(x.numpy())
                    df[self.sensitive_feature] = sensitive_feature[self.batch_size:].to(self.device)
                    df['y'] = y.numpy()
                    loss = self.loss(output, y, df, self.sensitive_feature, self.model)
                else:
                    loss = self.loss(output, y)
                loss.backward()

                 # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        self.model.train()

        (x, y, sensitive_feature) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        if isinstance(self.model, BinaryLogisticRegression):
            output = output.squeeze().to(torch.float)
            y = y.to(torch.float)
        if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                or isinstance(self.loss, HybridFairnessLoss)):
            df = pd.DataFrame(x.numpy())
            df[self.sensitive_feature] = sensitive_feature.to(self.device)
            df['y'] = y.numpy()
            loss = self.loss(output, y, df, self.sensitive_feature, self.model)
        else:
            loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_metrics(self, model=None):
        trainloader = self.load_train_data(self.batch_size*2)
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        for X, Y, sensitive_feature in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                    or isinstance(self.loss, HybridFairnessLoss)):
                df = pd.DataFrame(x[:self.batch_size].numpy())
                df[self.sensitive_feature] = sensitive_feature[:self.batch_size].to(self.device)
                df['y'] = y.numpy()
                loss = self.loss(output, y, df, self.sensitive_feature, self.model)
            else:
                loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                    or isinstance(self.loss, HybridFairnessLoss)):
                df = pd.DataFrame(x.numpy())
                df[self.sensitive_feature] = sensitive_feature[self.batch_size:].to(self.device)
                df['y'] = y.numpy()
                loss1 = self.loss(output, y, df, self.sensitive_feature, self.model)
            else:
                loss1 = self.loss(output, y)
            train_num += y.shape[0]
            losses += loss1.item() * y.shape[0]
        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y, sensitive_feature) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            output = self.model(x)
            if isinstance(self.model, BinaryLogisticRegression):
                output = output.squeeze().to(torch.float)
                y = y.to(torch.float)
            if (isinstance(self.loss, IndividualFairnessLoss) or isinstance(self.loss, GroupFairnessLoss)
                    or isinstance(self.loss, HybridFairnessLoss)):
                df = pd.DataFrame(x.numpy())
                df[self.sensitive_feature] = sensitive_feature[self.batch_size:].to(self.device)
                df['y'] = y.numpy()
                loss = self.loss(output, y, df, self.sensitive_feature, self.model)
            else:
                loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test_metrics1(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                pred_label = output.argmax(dim=1, keepdim=True)

                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    test_acc += (torch.sum((output >= 0.5) == y)).item()
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                if isinstance(self.model, BinaryLogisticRegression):
                    y_true.append(y.detach().cpu().numpy())
                else:
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auroc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auprc = metrics.average_precision_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auroc, auprc
