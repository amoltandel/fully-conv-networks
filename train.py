import copy

import torch
from torch.autograd import Variable


class Trainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            optimizer,
            criterion,
            scheduler,
            use_gpu=False

        ):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = model.cuda() if self.use_gpu else model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler


    def train(self, num_epochs=300, print_every=1000, save_every=10):
        print("Training started!")
        for epoch in range(1, 1 + num_epochs):
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            train_batch_loss = 0.0
            val_batch_loss = 0.0
            self.scheduler.step()
            for idx in range(len(self.train_dataset)):
                self.optimizer.zero_grad()
                sample = self.train_dataset[idx]
                input = Variable(sample['input'])
                target = Variable(sample['target'])
                if self.use_gpu:
                    input = input.cuda()
                    target = target.cuda()
                prediction = self.model(input)
                prediction = prediction.contiguous().view(-1, prediction.size(1))
                target = target.contiguous().view(-1)
                loss = self.criterion(prediction, target)

                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.data[0]
                train_batch_loss += loss.data[0]
                if idx % print_every == 0:
                    val_batch_loss = self.evaluate()
                    train_batch_loss /= print_every
                    print("Epoch: {:d} | Example: {:d} | Batch Training Loss: {:.5f} | Batch Validation Loss: {:.5f}".format(epoch, idx, train_batch_loss, val_batch_loss))
                    train_batch_loss = 0.0
            epoch_val_loss = self.evaluate()
            epoch_train_loss /= len(self.train_dataset)
            print("Epoch: {:d} | Training Loss: {:.5f} | Validation Loss: {:.5f}".format(epoch, epoch_train_loss, epoch_val_loss))
            if epoch % save_every == 0:
                best_model_wts = copy.deepcopy(self.model.cpu().state_dict())
                torch.save(best_model_wts, 'trained_models/'+self.model.model_name+'/epoch{:d}'.format(epoch) + '.pt')

    def evaluate(self):
        loss = 0.0
        for idx in range(len(self.val_dataset)):
            sample = self.train_dataset[idx]
            input = Variable(sample['input'])
            target = Variable(sample['target'])
            if self.use_gpu:
                input = input.cuda()
                target = target.cuda()
            prediction = self.model(input)
            prediction = prediction.contiguous().view(-1, prediction.size(1))
            target = target.contiguous().view(-1)
            loss += self.criterion(prediction, target).data[0]
        return loss / len(self.val_dataset)
