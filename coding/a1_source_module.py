import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, StatScores

import os
import math
from scipy.stats import entropy
import clip

from utils import HScore, CrossEntropyLabelSmooth, CustomLRScheduler
from networks import CLIPBlock, Resnet, FeatureExtractor, Classifier

class SourceModule(L.LightningModule):
    def __init__(self, datamodule, rejection_threshold=0.5, feature_dim=256, lr=1e-2, source_train_type='smooth',backbone = 'CLIP'):
        super(SourceModule, self).__init__()

        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num
        
        if backbone == 'CLIP':
            self.backbone = CLIPBlock()
        elif backbone == 'Resnet':
            self.backbone = Resnet()
        self.feature_extractor = FeatureExtractor(self.backbone.output_dim, feature_dim,type='bn')
        self.classifier = Classifier(self.known_classes_num, type='wn')


        self.lr = lr
        self.rejection_threshold = rejection_threshold
        if datamodule.category_shift == 'OPDA' or datamodule.category_shift == 'ODA':
            self.open_flag = True
        else:
            self.open_flag = False

        if source_train_type == 'smooth':
            self.train_loss = CrossEntropyLabelSmooth(num_classes=self.known_classes_num, epsilon=0.1, reduction=True)
        elif source_train_type == 'vanilla':
            self.train_loss = CrossEntropyLabelSmooth(num_classes=self.known_classes_num, epsilon=0.0, reduction=True)
        else:
            raise ValueError('Unknown source_train_type:', source_train_type)

        self.total_train_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num)

        self.total_test_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.test_statscores = StatScores(task='binary')
        self.test_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)
        self.domain = 'source'

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        iter_max = self.trainer.max_epochs * math.ceil(len(self.trainer.datamodule.train_set) /
                                                       self.trainer.datamodule.batch_size)
        optimizer = torch.optim.SGD(params_group)
        scheduler = CustomLRScheduler(optimizer, iter_max)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def lr_scheduler_step(self, scheduler, *args):
        scheduler.step(iter_num=self.global_step)

    def forward(self, x, apply_softmax=True):
        x = self.backbone(x)
        feature_embed = self.feature_extractor(x)
        x = self.classifier(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed

    def training_step(self, train_batch):
        x, y = train_batch
        y_hat, _ = self.forward(x, apply_softmax=True)
        # transform y to one-hot encoding
        onehot_label = torch.zeros_like(y_hat).scatter(1, y.unsqueeze(1), 1)
        loss = self.train_loss(y_hat, onehot_label)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.total_train_acc(y_hat, y)
        self.log('total_train_acc', self.total_train_acc, on_epoch=True, prog_bar=True)
        return loss

    def generate_class_prototypes(self):
        aggregated_class_features = torch.zeros(self.known_classes_num, self.feature_extractor.feature_dim)
        class_sample_counter = torch.zeros(self.known_classes_num)

        for x, y in self.trainer.datamodule.train_dataloader():
            with torch.no_grad():
                _, feature_embedding = self.forward(x)
                feature_embedding = feature_embedding.cpu()
                for c in range(self.known_classes_num):
                    idx = torch.where(y == c)
                    aggregated_class_features[c] += feature_embedding[idx].sum(dim=0)
                    class_sample_counter[c] += len(idx[0])

        return aggregated_class_features / torch.unsqueeze(class_sample_counter, -1)

    def on_train_end(self):
        print('Generating source prototypes...')
        prototypes = self.generate_class_prototypes()
        print('Save checkpoint...')
        os.makedirs(os.path.join(self.trainer.log_dir, 'checkpoints'))
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'class_prototypes': prototypes,
        }, self.trainer.log_dir + '/checkpoints/source_ckpt.pt')

    def test_step(self, test_batch, batch_idx):
        print('test step running')
        x, y = test_batch
        y = torch.where(y >= self.known_classes_num, self.known_classes_num, y)
        y_hat, feature_embedding = self.forward(x, apply_softmax=True)

        y_hat_entropy = torch.tensor(entropy(y_hat.cpu(), axis=1) / math.log(self.known_classes_num))
        pred = torch.where(y_hat_entropy <= self.rejection_threshold, torch.argmax(y_hat.cpu(), dim=1),
                           self.known_classes_num).to(self.device)
        self.total_test_acc(pred, y)
        self.log('total_test_acc', self.total_test_acc, on_step=False, on_epoch=True)

        if self.open_flag:
            self.test_hscore.update(pred, y)

            # calculate stat scores of rejection (number of TPs, FPs, TNs and FNs)
            rej_target = torch.where(y == self.known_classes_num, 1, 0)
            rej_pred = torch.where(pred == self.known_classes_num, 1, 0)
            self.test_statscores.update(rej_pred, rej_target)

        #self.test_feature_embeddings = torch.cat([self.test_feature_embeddings, feature_embedding.cpu()], 0)
        #self.test_labels = torch.cat([self.test_labels, y.cpu()], 0)

    def on_test_epoch_end(self):
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.test_hscore.compute()
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)
            
            
            
            

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

import wandb
import datetime
from datasets import DomainNetDataModule 


def main():
    Run_Name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    wandb.init(name= Run_Name)
    seed = 2816631403
    torch.manual_seed(seed)
    

    data = DomainNetDataModule(
        batch_size=64,
        category_shift='OPDA',
        train_domain = 'painting',
        test_domain = 'real'
    )
    
    
    model = SourceModule(datamodule = data, lr = 1e-2, backbone = wandb.config.backbone)

    trainer = Trainer(
        accelerator = 'auto',
        strategy = 'auto',
        devices = 'auto',
        num_nodes = 1,
        precision = 32,
        logger = WandbLogger(name=Run_Name, project="define unknown class for online SF-UniDA"),
        log_every_n_steps = 5,
        max_epochs = 20,#20
        min_epochs = 0,
        check_val_every_n_epoch = 1,
        enable_checkpointing=False,
        callbacks=[]
    )

    trainer.fit(model, datamodule=data)

    
sweep_config = {
    "method":"grid",
    "name" : "source_training",
    "parameters":{
        "backbone" :{"values": ['Resnet']}
    },
}

sweep_id = wandb.sweep(sweep = sweep_config, project = "define unknown class for online SF-UniDA")

print(sweep_id)
wandb.agent(sweep_id = sweep_id, function = main)
wandb.finish()