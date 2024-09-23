import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from ..trainmodel.models import BinaryLogisticRegression
from utils.fairness_loss import IndividualFairnessLoss, GroupFairnessLoss, HybridFairnessLoss


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self, new_local_epochs):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs # self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
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
                    df = pd.DataFrame(x[:self.batch_size].numpy())
                    df[self.sensitive_feature] = sensitive_feature[:self.batch_size].to(self.device)
                    df['y'] = y.numpy()
                    loss = self.loss(output, y, df, self.sensitive_feature, self.model)
                else:
                    loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
