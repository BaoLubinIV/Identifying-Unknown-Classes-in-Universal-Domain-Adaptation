import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import os
import math
import numpy as np
from scipy.stats import entropy
from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm

from networks import CLIPBlock
from utils import SupConLoss, HScore, OnlineUnknownClusteringMetric
from augmentation import get_tta_transforms
from dict_gen import generate_label_candidates


class Uni_Adapter(L.LightningModule):
    def __init__(self, datamodule, confidence_threshold =0.5, confidence_unknown_threshold =0.5, unknown_margin = 0.5, lr=1e-4, ckpt_dir = '', alpha = 0.95, train_adapter=''):
        super(Uni_Adapter, self).__init__()
        
        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num
        self.datamodule = datamodule
        self.ckpt_dir = ckpt_dir
        self.clip_backbone = CLIPBlock()
        self.adapter = nn.Linear(512, 512, bias=True)
        self._init_adapter()
        self.confidence_threshold = confidence_threshold
        self.confidence_unknown_threshold = confidence_unknown_threshold
        self.unknown_margin = unknown_margin
        
        self.step_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.total_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)
        self.unknown_cluster_eval = OnlineUnknownClusteringMetric()
        self.domain = 'target'
        self.alpha = alpha
        self.train_adapter = train_adapter
        self.lr = lr
        
        self.total_correct_unknown = 0
        self.total_unknown_samples = 0
                  
        
        if datamodule.category_shift == 'OPDA' or datamodule.category_shift == 'ODA':
            self.open_flag = True
        else:
            self.open_flag = False
       
    def _init_adapter(self):
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(512))  
            self.adapter.bias.copy_(torch.zeros(512)) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
        
    def on_train_start(self):
        print('prepare dict')
        known_classes = self.datamodule.known_classes_text  #len 200 label text of known classes
        
        self.target_classes_text = self.datamodule.all_classes_text

        augmented_labels = ["this is a picture of " + label for label in known_classes]


        with torch.no_grad():
            tokenized_texts = self.clip_backbone.tokenize(augmented_labels).to(self.device)
            text_features = self.clip_backbone.encode_text(tokenized_texts)  # [num_classes, 512]

        self.class_feature_dict = {label: feature for label, feature in zip(known_classes, text_features)}
        
        if not os.path.exists("candidates/candidates_dict_20000.npz"):
            generate_label_candidates(limit=10000)
        loaded_data = np.load("candidates/candidates_dict_20000.npz", allow_pickle=True)
        self.unknown_candidates_dict = loaded_data['candidates_dict'].item()
        
        
        
    def on_train_epoch_end(self):
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_hscore.compute()
            print(f"H-Score: {h_score}")
            print(f"Known Accuracy: {known_acc}")
            print(f"Unknown Accuracy: {unknown_acc}")
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)
        
        unknown_label_acc = self.total_correct_unknown / self.total_unknown_samples
        self.log("unknown_label_acc", unknown_label_acc, prog_bar=True)
        
        unknown_metrics = self.unknown_cluster_eval.compute()
        self.log("precision", unknown_metrics["Precision"])
        self.log("recall", unknown_metrics["Recall"])
        self.log("f1_score", unknown_metrics["F1-Score"])
        self.log("noise_ratio", unknown_metrics["Noise Ratio"])

    def on_train_end(self):
        pass

    def training_step(self, train_batch):
        x, y_true = train_batch
        y = torch.where(y_true >= self.known_classes_num, self.known_classes_num, y_true).long()

        img_features = self.clip_backbone.forward(x)
        adapted_features = self.adapter(img_features)
        
        
        candidate_features = torch.stack(
            [torch.tensor(v, dtype=torch.float32) for v in self.unknown_candidates_dict.values()]
        ).to(self.device) # (num_candidates, 512)
        candidate_features = self.adapter(candidate_features)
        
        class_features = torch.stack(list(self.class_feature_dict.values())).to(self.device)  # (num_classes, 512)
        class_features = self.adapter(class_features)
        
        cosine_similarity = torch.matmul(adapted_features, class_features.T)  # (batch_size, num_classes)
        cosine_similarity = cosine_similarity / (adapted_features.norm(dim=1, keepdim=True) * class_features.norm(dim=1)+1e-8)
        
        
        max_sim_known, preds = torch.max(cosine_similarity, dim=1) 
        
        
        rejected_mask = max_sim_known < self.confidence_threshold 
        confident_mask = max_sim_known >= self.confidence_threshold
        

        ##——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        #unknown detection        
        if rejected_mask.any():  
            rejected_features = adapted_features[rejected_mask]
            known_sim_for_rejected = max_sim_known[rejected_mask]
            
            cosine_similarity_candidate = torch.matmul(rejected_features, candidate_features.T)  # (#rejected, # candidates)
            cosine_similarity_candidate = cosine_similarity_candidate / (
                rejected_features.norm(dim=1, keepdim=True) * candidate_features.norm(dim=1)+1e-8
            )

            best_candidate_sim, best_candidate_index = torch.max(cosine_similarity_candidate, dim=1)  # (#rejected, 1)assigned label number
            
            #print(f"this should be lehgth of not confident known: {len(best_candidate_index)}")
            # unknown confirmation
            keep_known = max_sim_known[rejected_mask] >= best_candidate_sim * self.alpha # (#rejected, 1)
            keep_known_mask = torch.zeros_like(rejected_mask, dtype=torch.bool) # (#batch_size,1)
            keep_known_mask[rejected_mask] = keep_known 
            
            preds[rejected_mask & ~keep_known_mask] = self.known_classes_num  # confirmed unknown with same single unknown class
            
            
            selected_best_candidate_index = best_candidate_index[keep_known == False]  # (#confirmed unknown,1)
            #print(f"this should be assigned label number for confirmed unknown: {selected_best_candidate_index}")

            #print(f"this should be length of confirmed unknown: {len(selected_best_candidate_index)}")
            
            ##——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            #evaluate for label acc
            #ground truth label text of detected unknown samples
            true_unknown_labels = [self.target_classes_text[int(idx)] for idx in y_true[rejected_mask & ~keep_known_mask].tolist()]
            #print(f"this should be true label text: {true_unknown_labels}")
            #print(len(true_unknown_labels))

            # assigned label text of detected unknown samples
            assigned_labels = [list(self.unknown_candidates_dict.keys())[idx.item()] for idx in selected_best_candidate_index]
            #print(f"this should be assigned label text: {assigned_labels}")
            #print(len(assigned_labels))
            
            correct_matches = sum(1 for gt, assigned in zip(true_unknown_labels, assigned_labels) if gt == assigned)
            batch_unknown_count = len(true_unknown_labels)

            self.total_correct_unknown += correct_matches
            self.total_unknown_samples += batch_unknown_count

            batch_unknown_acc = correct_matches / batch_unknown_count if batch_unknown_count > 0 else 0.0
            self.log("batch_unknown_acc", batch_unknown_acc, on_epoch=True, prog_bar=True)
                
                
                
                
            #evaluate for unknown clusters precision and recall
            y_true_unknown = y_true[rejected_mask & ~keep_known_mask]
            self.unknown_cluster_eval.update(selected_best_candidate_index, y_true_unknown)
            self.log("precision_step", self.unknown_cluster_eval.latest_precision, on_step=True)
            self.log("recall_step", self.unknown_cluster_eval.latest_recall, on_step=True)
            self.log("noisy_ratio_step", self.unknown_cluster_eval.latest_noise_ratio, on_step=True)
        
        
        
        #evaluate for known acc and unknown acc
        #print('preds after unknown sample detection:', preds)
        self.step_acc(preds, y)
        self.total_hscore.update(preds, y)
        self.log("step_acc", self.step_acc, on_epoch=True, prog_bar=True)
        
        
        

        
        
        #Adaptation
        if self.train_adapter == 'cosine_similarity_loss' and rejected_mask.any():
            confident_unknown_mask = best_candidate_sim >= self.confidence_unknown_threshold # (#rejected,1)
            if confident_mask.any():
                valid_confident_mask = confident_mask & (y < self.known_classes_num)
                confident_adapted_features = adapted_features[valid_confident_mask]
                confident_target_features = class_features[y[valid_confident_mask]]

                cosine_sim = torch.sum(confident_adapted_features * confident_target_features, dim=1) / (
                    torch.norm(confident_adapted_features, dim=1) * torch.norm(confident_target_features, dim=1) + 1e-8
                )
                loss_known = 1 - cosine_sim.mean()
            else:
                loss_known = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(self.device) 
            if confident_unknown_mask.any():
                confident_unknown_features = rejected_features[confident_unknown_mask]
                confident_unknown_target = candidate_features[best_candidate_index[confident_unknown_mask]] 

                cosine_sim_unknown = torch.sum(confident_unknown_features * confident_unknown_target, dim=1) / (
                    torch.norm(confident_unknown_features, dim=1) * torch.norm(confident_unknown_target, dim=1) + 1e-8
                )
                loss_unknown = 1 - cosine_sim_unknown.mean()
            else:
                loss_unknown = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(self.device) 
            loss = loss_known + loss_unknown

        # another adaption solution: contrasitive learning
        elif self.train_adapter == 'contrasitive_loss' and rejected_mask.any():
            
            confident_known_mask = confident_mask & (y < self.known_classes_num)
            confident_unknown_mask = rejected_mask & (max_sim_known < self.confidence_unknown_threshold)
            
            loss_list = []
            
            
            if confident_known_mask.any():
                known_features = adapted_features[confident_known_mask]
                known_targets = class_features[y[confident_known_mask]]
                cos_sim_known = torch.nn.functional.cosine_similarity(known_features, known_targets, dim=1)
                loss_known = (1 - cos_sim_known).mean()
                print('loss known:',loss_known)
                loss_list.append(loss_known)
            
            if confident_unknown_mask.any():
                unknown_features = adapted_features[confident_unknown_mask]
                
                cos_sim_unknown_all = torch.nn.functional.cosine_similarity(
                    unknown_features.unsqueeze(1), class_features.unsqueeze(0), dim=2
                ) #(#unknown_samples, #known_classes)
                # keep unknown_margin as decision boundary
                max_sim_unknown, _ = cos_sim_unknown_all.max(dim=1)
                loss_unknown = torch.relu(max_sim_unknown - self.unknown_margin).mean()
                print('loss_unknown:', loss_unknown)
                loss_list.append(loss_unknown)
            
            if loss_list:
                loss = sum(loss_list)
            else:
                loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(self.device)
        else:
            loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(self.device)
        

        return loss
        
        


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
    
    
    model = Uni_Adapter(
        datamodule = data, 
        lr = 1e-6, 
        confidence_threshold = wandb.config.confidence_threshold,
        confidence_unknown_threshold = wandb.config.confidence_unknown_threshold,
        unknown_margin = wandb.config.unknown_margin,
        alpha = wandb.config.alpha,
        train_adapter = wandb.config.train_adapter
        
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
    "name" : "source_training",
    "parameters":{
        "confidence_threshold" :{"values": [0.35]},
        "confidence_unknown_threshold" :{"values": [0.4]},
        "unknown_margin" :{"values": [0.15]},
        "alpha" :{"values": [0.95]},
        "train_adapter" :{"values": ['','cosine_similarity_loss']}#'cosine_similarity_loss' or 'contrasitive_loss' or ''
    },
}

sweep_id = wandb.sweep(sweep = sweep_config, project = "define unknown class for online SF-UniDA")

print(sweep_id)
wandb.agent(sweep_id = sweep_id, function = main)
wandb.finish()
