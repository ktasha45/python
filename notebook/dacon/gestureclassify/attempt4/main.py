import torch
import pandas as pd
import numpy as np
import os
import tqdm
import pytorch_lightning as pl
import torchmetrics
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.utils import data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PIL import Image

class make_dataset(data.Dataset):
  def __init__(self, idx_list, file, mode):
    self.file = file
    self.idx_list = idx_list
    self.mode = mode

  def __len__(self):
    return len(self.idx_list)

  def __getitem__(self, index):
    img = self.file[index, 1:33].reshape(1, -1, 4).astype('float32')
    img2 = np.zeros((1, 8,4)).astype('float32')
    for i in range(4):
      img2[:, i] = img[:, 4-i]

    if self.mode == 'train':
      label = self.file[index, 33].astype('int64')
      return np.concatenate([img, img2], axis=0), label
    else:
      return np.concatenate([img, img2], axis=0)

def make_dataloader(batch_size=64, valid_size=0.2):
    traindata = pd.read_csv('../data/train.csv').to_numpy()
    testdata = pd.read_csv('../data/test.csv').to_numpy()

    num_train = len(traindata)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = np.int32(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    test_idx = list(range(len(testdata)))

    train_set = make_dataset(train_idx, traindata, 'train')
    valid_set = make_dataset(valid_idx, traindata, 'train')
    test_set = make_dataset(test_idx, testdata, 'test')

    train_loader = data.DataLoader(train_set, batch_size=batch_size)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size)
    test_loader = data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader, valid_loader

class network(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.net = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(128,256,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(512,512,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_fn(pred, target)
        acc = torchmetrics.functional.accuracy(pred, target)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main():
    print(torch.__version__)
    print(print(torch.cuda.get_device_name(0)))

    train_csv = pd.read_csv('../data/train.csv')
    test_csv = pd.read_csv('../data/test.csv')

    batch_size = 64
    train_loader, test_loader, valid_loader = make_dataloader(batch_size)

    print(train_loader.dataset.__len__())
    print(valid_loader.dataset.__len__())

    print(train_loader.dataset.__getitem__(0)[0][0].shape)

    fig = plt.figure(figsize=(12, 24))
    for i, (imgs, labels) in enumerate(train_loader):
        print(imgs.shape)
        break
        # for j in range(9):
        #     ax = fig.add_subplot(3, 3, j + 1)
        #     ax.set_xlabel(labels[j].numpy())
        #     plt.imshow(imgs[j][0])
        # break

    model = network()
    print(model.net)

    callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=False)]
    trainer = pl.Trainer(max_epochs=200, gpus=1, callbacks=callbacks)
    trainer.fit(model, train_loader, valid_loader)
    

    sample_submission = pd.read_csv('../data/sample_submission.csv')
    batch_index = 0
    for i, data in enumerate(test_loader):
        outputs = model.forward(data)
        batch_index = i * batch_size
        max_vals, max_indices = torch.max(outputs, 1)
        sample_submission.iloc[batch_index:batch_index + batch_size, 1:] = max_indices.long().cpu().numpy()[:,np.newaxis]
    sample_submission.to_csv('version7.csv', index=False)

if __name__=='__main__':
    main()