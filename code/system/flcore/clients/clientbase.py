import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import itertools
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from ..trainmodel.models import BinaryLogisticRegression
from utils.fairness_loss import IndividualFairnessLoss, GroupFairnessLoss, HybridFairnessLoss, fairness_metrics


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.datapath = args.datapath
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        if not args.use_fairness_penalty:
            if isinstance(self.model, BinaryLogisticRegression):
                self.loss = nn.BCELoss() # notes from Siqi:
                # add group and individual fairness from paper: Berk R, Heidari H, Jabbari S, Joseph M, Kearns M, Morgenstern J, Neel S, Roth A. A convex framework for fair regression. arXiv preprint arXiv:1706.02409. 2017 Jun 7.
                # three options: 1) individual fairness only 2) group fairness 3) hybrid
                # details: 1) consider CV for lambda selection. 2) details of optimization need further discussion, choice of initial values, training loops etc.;
            else:
                self.loss = nn.CrossEntropyLoss()
        else:
            self.fairness_lambda = args.fairness_lambda
            self.fairness_gamma = args.fairness_gamma
            if args.fairness_loss == 'individual':
                self.loss = IndividualFairnessLoss(fairness_lambda=self.fairness_lambda, L2_gamma=self.fairness_gamma)
            elif args.fairness_loss == 'group':
                self.loss = GroupFairnessLoss(fairness_lambda=self.fairness_lambda, L2_gamma=self.fairness_gamma)
            elif args.fairness_loss == 'hybrid':
                self.loss = HybridFairnessLoss(fairness_lambda=self.fairness_lambda, L2_gamma=self.fairness_gamma)
            else:
                print(f'{args.fairness_loss} fairness loss not implemented, default to BCE loss.')
                self.loss = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        if 'adult' in self.dataset:
            self.sensitive_feature = 'gender'
        elif 'MIMIC' in self.dataset:
            self.sensitive_feature = 'Race'
        elif 'US_homo' in self.dataset or 'US_hete' in self.dataset or 'USSG' in self.dataset:
            self.sensitive_feature = 'SEX'


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.datapath,self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.datapath,self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        y_pred = []
        sensitive_feature_all = []
        X_all = []
        
        with torch.no_grad():
            for x, y, sensitive_feature in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    test_acc += (torch.sum((output >= 0.5) == y)).item()
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                X_all.append(x.detach().cpu().numpy())
                if output.dim() == 0:
                    y_prob.append(np.array(output.unsqueeze(0).detach().cpu().numpy()))
                    y_pred.append(np.array((output >= 0.5).unsqueeze(0).detach().cpu().numpy()))
                else:
                    y_prob.append(output.detach().cpu().numpy())
                    y_pred.append((output >= 0.5).detach().cpu().numpy())
                sensitive_feature_all.append(sensitive_feature.detach().cpu().numpy())
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

        X_all = np.concatenate(X_all)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        sensitive_feature_all = np.concatenate(sensitive_feature_all, axis=0)

        auroc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auprc = metrics.average_precision_score(y_true, y_prob, average='micro')
        MSE = metrics.mean_squared_error(y_true, y_prob)
        fairness = fairness_metrics(y_true, y_pred, sensitive_feature_all, X_all)

        return test_acc, test_num, auroc, auprc, MSE, fairness

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y, sensitive_feature in trainloader:
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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics1()
        print(stats)
        # stats_train = self.train_metrics()

        #test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # test_auroc = sum(stats[3])*1.0 / sum(stats[1])
        # test_auprc = sum(stats[4])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aurocs = [a / n for a, n in zip(stats[3], stats[1])]
        # auprcs = [a / n for a, n in zip(stats[4], stats[1])]
        
        # if acc == None:
        #     self.rs_test_acc.append(test_acc)
        # else:
        #     acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        # print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        # print("Averaged Test AUROC: {:.4f}".format(test_auroc))
        # print("Averaged Test AUPRC: {:.4f}".format(test_auprc))
        # # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUROC: {:.4f}".format(np.std(aurocs)))
        # print("Std Test AUPRC: {:.4f}".format(np.std(auprcs)))
    def test_metrics1(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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

                if isinstance(self.model, BinaryLogisticRegression):
                    output = output.squeeze().to(torch.float)
                    test_acc += (torch.sum((output >= 0.5) == y)).item()
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                # print('output:\n', output)
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
                # print('truth:\n', y.detach().cpu().numpy())

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auroc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auprc = metrics.average_precision_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auroc, auprc
