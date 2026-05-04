import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import os
import math
from scipy.stats import entropy
from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm

from networks import CLIPBlock, Resnet, FeatureExtractor, Classifier
from utils import SupConLoss, HScore, OnlineUnknownClusteringMetric
from augmentation import get_tta_transforms

from sklearn.cluster import DBSCAN
import numpy as np


class COMET(L.LightningModule):
    def __init__(self, datamodule, rejection_threshold=0.5, feature_dim=256, lr=1e-4, lower_confidence_threshold=0.25,
                 upper_confidence_threshold=0.75, ckpt_dir='', cl_projection_dim=128, cl_temperature=0.1,
                 m_teacher_momentum=0.999, lbd=0.1, use_source_prototypes=True, dbscan_eps=1, dbscan_min_samples=3,
                 backbone = 'CLIP',learn_from_unknown = False):
        super(COMET, self).__init__()


        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        if backbone == 'CLIP':
            self.backbone = CLIPBlock()
        elif backbone == 'Resnet':
            self.backbone = Resnet()
        self.feature_extractor = FeatureExtractor(self.backbone.output_dim, feature_dim,type='bn')
        self.classifier = Classifier(self.known_classes_num, type='wn')

        if ckpt_dir != '':
            checkpoint = torch.load(ckpt_dir, map_location=torch.device('cpu'))
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        self.lr = lr
        self.rejection_threshold = rejection_threshold

        if datamodule.category_shift == 'OPDA' or datamodule.category_shift == 'ODA':
            self.open_flag = True
        else:
            self.open_flag = False


        self.lower_confidence_threshold = lower_confidence_threshold
        self.upper_confidence_threshold = upper_confidence_threshold

        
        self.feature_extractor_teacher = self.copy_model(self.feature_extractor)
        self.classifier_teacher = self.copy_model(self.classifier)

        self.ckpt_dir = ckpt_dir
        self.class_prototypes = None
        self.prototype_sum = torch.zeros(self.known_classes_num, feature_dim)
        self.prototype_sample_counter = torch.zeros(self.known_classes_num, 1)

        self.total_online_tta_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.total_online_tta_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

        cl_projector = nn.Sequential(nn.Linear(self.feature_extractor.feature_dim, cl_projection_dim),
                                     nn.ReLU(), nn.Linear(cl_projection_dim, cl_projection_dim)).to(self.device)
        self.contrastive_loss = SupConLoss(projector=cl_projector, temperature=cl_temperature)
        self.tta_transform = get_tta_transforms()
        self.m_teacher_momentum = m_teacher_momentum
        self.lbd = lbd

        self.use_source_prototypes = use_source_prototypes
        self.automatic_optimization = False
        
        self.noisy_unknown_samples = np.empty((0, feature_dim))
        self.unknown_cluster_prototypes = np.empty((0, feature_dim))
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        self.unknown_cluster_eval = OnlineUnknownClusteringMetric()
        self.con_loss_unknown = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.learn_from_unknown = learn_from_unknown
        self.domain = 'target'

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.contrastive_loss.projector.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        return optimizer
        
    def forward(self, x, apply_softmax=True, tta_transforms=False):
        x = self.backbone(x,tta_transforms = tta_transforms)
        feature_embed = self.feature_extractor(x)
        x = self.classifier(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed
        
    def on_train_start(self):
        if torch.cuda.is_available():
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cuda'))['class_prototypes']
        else:
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cpu'))['class_prototypes']
        self.unknown_cluster_eval.to(self.device)

    def generate_pseudo_labels(self, y_hat):
        y_hat_entropy = torch.tensor(entropy(y_hat.detach().cpu(), axis=1) / math.log(self.known_classes_num))
        confident_idx = torch.where(torch.logical_or(y_hat_entropy <= self.lower_confidence_threshold,
                                                     y_hat_entropy >= self.upper_confidence_threshold))[0]
        pseudo_labels = torch.where(y_hat_entropy[confident_idx] >= self.upper_confidence_threshold,
                                    self.known_classes_num, torch.argmax(y_hat.cpu(), dim=1)[confident_idx])
        
        num_unknown = torch.sum(pseudo_labels == self.known_classes_num).item()
        print(f"Number of samples labeled as 'unknown': {num_unknown}")
        return confident_idx, pseudo_labels

    def forward_teacher(self, x, apply_softmax=True):
        x = self.backbone(x)
        feature_embed = self.feature_extractor_teacher(x)
        x = self.classifier_teacher(feature_embed)
        if apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x, feature_embed

    def copy_model(self, model):
        if not isinstance(model, FeatureExtractor):  # https://github.com/pytorch/pytorch/issues/28594
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model

    def on_train_epoch_end(self):
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_online_tta_hscore.compute()
            print(f"H-Score: {h_score}")
            print(f"Known Accuracy: {known_acc}")
            print(f"Unknown Accuracy: {unknown_acc}")
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)
            
            unknown_metrics = self.unknown_cluster_eval.compute()
            self.log("precision", unknown_metrics["Precision"])
            self.log("recall", unknown_metrics["Recall"])
            self.log("f1_score", unknown_metrics["F1-Score"])
            self.log("noise_ratio", unknown_metrics["Noise Ratio"])

    def on_train_end(self):
        '''
        os.makedirs(os.path.join(self.trainer.log_dir, 'checkpoints'))
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
        }, self.trainer.log_dir + '/checkpoints/adapted_ckpt.pt')
        '''

    def training_step(self, train_batch):
        opt = self.optimizers()
        self.on_test_model_eval()
        self.class_prototypes = self.class_prototypes.to(self.device)
        self.prototype_sum = self.prototype_sum.to(self.device)
        self.prototype_sample_counter = self.prototype_sample_counter.to(self.device)

        x, y_true = train_batch
        y = torch.where(y_true >= self.known_classes_num, self.known_classes_num, y_true)

        opt.zero_grad()

        # FORWARD
        y_hat, features = self.forward(x, apply_softmax=True)
        y_hat_aug, features_aug = self.forward(x, tta_transforms=True)
        
        y_hat_teacher, _ = self.forward_teacher(x, apply_softmax=True)

        # ADAPTATION
        with torch.no_grad():
            pseudo_label_idx, pseudo_label = self.generate_pseudo_labels(y_hat_teacher)
            pseudo_label = pseudo_label.to(self.device)
            pseudo_label_idx = pseudo_label_idx.to(self.device)
        known_idx = torch.where(pseudo_label != self.known_classes_num)[0].to(self.device)
        print(f"Number of samples labeled as 'known': {len(known_idx)}")

        if not self.use_source_prototypes:
            with torch.no_grad():
                self.prototype_sum[pseudo_label[known_idx]] += features[pseudo_label_idx[known_idx]]
                for label in pseudo_label[known_idx]:
                    self.prototype_sample_counter[label] += 1

                self.class_prototypes = self.prototype_sum / (self.prototype_sample_counter + 1e-5)
                self.class_prototypes = self.class_prototypes.to(self.device)

        cl_known_features = torch.cat([torch.unsqueeze(self.class_prototypes[pseudo_label[known_idx]], dim=1),
                                       torch.unsqueeze(features[pseudo_label_idx[known_idx]], dim=1),
                                       torch.unsqueeze(features_aug[pseudo_label_idx[known_idx]], dim=1)],
                                      dim=1)
        unknown_idx = torch.where(pseudo_label == self.known_classes_num)[0]
        cl_unknown_features = torch.cat([torch.unsqueeze(features[pseudo_label_idx[unknown_idx]], dim=1),
                                         torch.unsqueeze(features_aug[pseudo_label_idx[unknown_idx]], dim=1)], dim=1)

        y_hat_entropy = -torch.matmul(y_hat, torch.log(y_hat.T)) / torch.log(torch.tensor(self.known_classes_num))
        y_hat_entropy = torch.diagonal(y_hat_entropy)
        
        

        if len(known_idx) != 0:
            con_loss = self.contrastive_loss(cl_known_features, labels=pseudo_label[known_idx],
                                             confident_unknown_features=cl_unknown_features)
            entropy_loss = y_hat_entropy[pseudo_label_idx[known_idx]].mean() -\
                           y_hat_entropy[pseudo_label_idx[unknown_idx]].mean()
            loss = con_loss + self.lbd * entropy_loss
            self.manual_backward(loss, retain_graph=True)
            self.log('tta_loss', loss, on_epoch=True, prog_bar=True)
        else:
            loss = None
        opt.step()
        opt.zero_grad()

        
        self.feature_extractor_teacher = self.update_ema_variables(ema_model=self.feature_extractor_teacher,
                                                                   model=self.feature_extractor,
                                                                   alpha_teacher=self.m_teacher_momentum)
        self.classifier_teacher = self.update_ema_variables(ema_model=self.classifier_teacher,
                                                            model=self.classifier,
                                                            alpha_teacher=self.m_teacher_momentum)

        # PREDICTION
        with torch.no_grad():
            pred = torch.where(y_hat_entropy.detach() <= self.rejection_threshold, torch.argmax(y_hat.detach(), dim=1),
                               self.known_classes_num).to(self.device)
            
            num_unknown = torch.sum(pred == self.known_classes_num).item()
            print(f"Number of samples predicted as 'unknown': {num_unknown}")
            
            self.total_online_tta_acc(pred, y)
            self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)
            if self.open_flag:
                self.total_online_tta_hscore.update(pred, y)
                
            y_true_unknown = y_true[torch.where(pred == self.known_classes_num)[0]]
                


        # Clustering for unknown
        print('clustering for unknown start')
        unknown_idx = torch.where(pred == self.known_classes_num)[0]
        print(unknown_idx)

        if len(unknown_idx) > 0:
            # prepare unknown features 
            unknown_features = features[unknown_idx].detach().cpu().numpy()
            combined_features = unknown_features
            repeated_prototypes = np.repeat(self.unknown_cluster_prototypes, self.dbscan_min_samples, axis=0)
            if self.unknown_cluster_prototypes.size != 0:                                
                combined_features = np.vstack((repeated_prototypes, combined_features))
            if self.noisy_unknown_samples.size != 0:
                combined_features = np.vstack((combined_features, self.noisy_unknown_samples))
            
            # DBSCAN for clustering
            print('start DBSCAN')
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples) 
            cluster_labels = dbscan.fit_predict(combined_features)
            
            unknown_start_idx = repeated_prototypes.shape[0]
            unknown_end_idx = unknown_start_idx + unknown_features.shape[0]
            y_hat_unknown = cluster_labels[unknown_start_idx:unknown_end_idx]
            
            # update noisy unknown samples
            noise_indices = np.where(cluster_labels == -1)[0]
            self.noisy_unknown_samples = combined_features[noise_indices]
            self.noisy_unknown_samples = self.noisy_unknown_samples[-5000:] 
            print('noisy unknown updated')
            
            # update unknown clusters prototypes 
            new_cluster_prototypes = []
            for cluster_id in set(cluster_labels) - {-1}:  
                cluster_points = combined_features[cluster_labels == cluster_id]
                cluster_center = cluster_points.mean(axis=0)
                new_cluster_prototypes.append(cluster_center)
            
            self.unknown_cluster_prototypes = np.array(new_cluster_prototypes)            
            print(f"Number of noisy samples: {len(self.noisy_unknown_samples)}")
            print(f"Number of cluster prototypes: {len(self.unknown_cluster_prototypes)}")
            
            #tp, fp, fn = self.unknown_cluster_eval.update(y_hat_unknown, y_true_unknown)
            #noisy_ratio_step = (y_hat_unknown == -1).sum().item()/len(y_hat_unknown)

            #precision_step = tp / (tp + fp) if (tp + fp) > 0 else 0
            #recall_step = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            y_hat_unknown = torch.tensor(y_hat_unknown, device=self.device)
            y_true_unknown = torch.tensor(y_true_unknown, device=self.device)
            self.unknown_cluster_eval.update(y_hat_unknown, y_true_unknown)
            self.log("precision_step", self.unknown_cluster_eval.latest_precision, on_step=True)
            self.log("recall_step", self.unknown_cluster_eval.latest_recall, on_step=True)
            self.log("noisy_ratio_step", self.unknown_cluster_eval.latest_noise_ratio, on_step=True)
            
            
            
            #learning for unknown clusters
            if isinstance(y_hat_unknown, torch.Tensor):
                y_hat_unknown = y_hat_unknown.detach().cpu().numpy()
            clustered_idx = np.where(y_hat_unknown != -1)[0]
            if len(clustered_idx) > 0 and self.learn_from_unknown:
                cluster_prototypes = torch.from_numpy(self.unknown_cluster_prototypes[y_hat_unknown[clustered_idx]]).to(self.device)
                clustered_features = features[unknown_idx][clustered_idx].to(self.device)

                mse_loss = torch.nn.functional.mse_loss(clustered_features, cluster_prototypes)

                self.manual_backward(mse_loss, retain_graph=True)
                self.log('mse_loss_unknown', mse_loss, on_epoch=True, prog_bar=True)

                opt.step()
                opt.zero_grad()
            

        else:
            print("No unknown samples detected.")

        






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
        batch_size=32,
        category_shift='OPDA',
        train_domain = 'painting',
        test_domain = 'real'
    )
    
    
    model = COMET(
        datamodule=data,  
        rejection_threshold=wandb.config.rejection_threshold,
        lr=wandb.config.lr,
        lower_confidence_threshold=wandb.config.lower_confidence_threshold,
        upper_confidence_threshold=wandb.config.upper_confidence_threshold,
        ckpt_dir= 'checkpoints/source_ckpt_painting_CLIP.pt',
        cl_projection_dim=128,
        cl_temperature=0.1,
        m_teacher_momentum=wandb.config.teacher_momentum,
        lbd=0.1,
        use_source_prototypes=wandb.config.use_source_prototypes,
        backbone = wandb.config.backbone,
        dbscan_eps = wandb.config.dbscan_eps,
        dbscan_min_samples = wandb.config.dbscan_min_samples,
        learn_from_unknown = wandb.config.learn_from_unknown
    )
    
    trainer = Trainer(
        accelerator = 'auto',
        strategy = 'auto',
        devices = 'auto',
        num_nodes = 1,
        precision = 32,
        logger = WandbLogger(name=Run_Name, project="define unknown class for online SF-UniDA"),
        log_every_n_steps = 5,
        max_epochs = 1,
        min_epochs = 0,
        check_val_every_n_epoch = 1,
        enable_checkpointing=False,
        callbacks=[]
    )

    trainer.fit(model, datamodule=data)

    
sweep_config = {
    "method":"grid",
    "name" : 'adaption',
    "parameters":{
        "rejection_threshold" : {"values":[0.6]},
        "lower_confidence_threshold" : {"values":[0.25]},
        "upper_confidence_threshold" : {"values":[0.75]},
        "use_source_prototypes" : {"values":[True]},
        "backbone" : {"values":['CLIP']},
        "lr" : {"values":[1e-4]},
        "teacher_momentum" : {"values":[0.999]},
        "dbscan_eps" : {"values":[6]},
        "dbscan_min_samples" : {"values":[4]},
        "learn_from_unknown" : {"values":[False,True]}
    },
}

sweep_id = wandb.sweep(sweep = sweep_config, project = "define unknown class for online SF-UniDA")

print(sweep_id)
wandb.agent(sweep_id = sweep_id, function = main)
wandb.finish()





