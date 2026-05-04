import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, iter_max, weight_decay=1e-3, momentum=0.9, nesterov=True):
        self.optimizer = optimizer
        self.iter_max = iter_max
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        super(CustomLRScheduler, self).__init__(optimizer)

    def step(self, iter_num=0, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / self.iter_max) ** (-power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            param_group['weight_decay'] = self.weight_decay
            param_group['momentum'] = self.momentum
            param_group['nesterov'] = self.nesterov
        return self.optimizer


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)

        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


# Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, projector, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

        self.projector = projector

    def forward(self, features, labels=None, mask=None, confident_unknown_features=torch.tensor([])):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            confident_unknown_features: features of samples labeled as unkown by pseudo-labeling
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        self.projector.to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number M of different views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # stack views to "batch" (size: M*N)
        contrast_feature = self.projector(contrast_feature)  # project into projection space
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits i.e. M*NxM*N-matrix of z_p*z_q/tau for all p,q in I
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if confident_unknown_features.numel() != 0:
            confident_unknown_features = torch.cat(torch.unbind(confident_unknown_features, dim=1), dim=0)
            confident_unknown_features = self.projector(confident_unknown_features)
            confident_unknown_features = F.normalize(confident_unknown_features, p=2, dim=1)
            # compute dot products of each known sample with all unknown samples, respectively
            confident_unknown_contrast = torch.div(torch.matmul(anchor_feature, confident_unknown_features.T),
                                                   self.temperature)
            confident_unknown_contrast = confident_unknown_contrast - logits_max.detach()[0]
            confident_unknown_contrast = torch.exp(confident_unknown_contrast)
        else:
            confident_unknown_contrast = torch.tensor([0])

        # tile mask, i.e. main diagonals of the M^2 NxN-submatrices
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # mask is 1 where z_i*z_j(i) in logits

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # exp(z_i*z_a/tau) for all a in A(i), 0 for a=i (on main diagonal)
        # compute log(exp(z_i*z_j(i)/tau)/sum_a exp(z_i*z_a/tau))=z_i*z_a/tau-log(sum_a exp(z_i*z_a/tau))
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + torch.ones_like(exp_logits_sum) * confident_unknown_contrast.sum())

        # compute mean of log-likelihood over positive for each i in I
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # compute mean over all i in I
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HScore(torchmetrics.Metric):
    def __init__(self, known_classes_num, shared_classes_num):
        super(HScore, self).__init__()

        self.total_classes_num = known_classes_num + 1
        self.shared_classes_num = shared_classes_num

        self.add_state("correct_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")
        self.add_state("total_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")
        
        self.add_state("tp_unknown", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_unknown", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn_unknown", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tn_unknown", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        for c in range(self.total_classes_num):
            self.total_per_class[c] += (target == c).sum()
            self.correct_per_class[c] += ((preds == target) * (target == c)).sum()
            
        unknown_class = self.total_classes_num - 1
        
        self.tp_unknown += ((preds == unknown_class) & (target == unknown_class)).sum().item()
        self.fp_unknown += ((preds == unknown_class) & (target != unknown_class)).sum().item()
        self.fn_unknown += ((preds != unknown_class) & (target == unknown_class)).sum().item()
        self.tn_unknown += ((preds != unknown_class) & (target != unknown_class)).sum().item()

    def compute(self):
        per_class_acc = self.correct_per_class / (self.total_per_class + 1e-5)
        known_acc = per_class_acc[:self.shared_classes_num].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
        
        print(f"Unknown Class Statistics:")
        print(f"  True Positives (TP): {self.tp_unknown}")
        print(f"  False Positives (FP): {self.fp_unknown}")
        print(f"  False Negatives (FN): {self.fn_unknown}")
        print(f"  True Negatives (TN): {self.tn_unknown}")
        
        return h_score, known_acc, unknown_acc








class OnlineUnknownClusteringMetric(torchmetrics.Metric):
    def __init__(self, ):
        super().__init__()

        # Register state variables

        self.add_state("noise_count", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        
        self.add_state("y_true", default=torch.empty(0, dtype=torch.long), persistent=False)
        self.add_state("y_hat", default=torch.empty(0, dtype=torch.long), persistent=False)
        
        self.add_state("latest_precision", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("latest_recall", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("latest_noise_ratio", default=torch.tensor(0.0), dist_reduce_fx=None)
        
        self.add_state("sum_precision", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("sum_recall", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("step_counter", default=torch.tensor(0.0), dist_reduce_fx=None)

        
    def update(self, y_hat, y_true):
        # Count total samples
        self.total_samples += y_true.numel()
        # Noise points
        self.noise_count += (y_hat == -1).sum().item()
        
        y_hat = torch.tensor(y_hat)
        y_true = torch.tensor(y_true)
        
        y_hat = y_hat.detach()
        y_true = y_true.detach()

        self.y_true = torch.cat((self.y_true, y_true), dim=0)[-200:]
        self.y_hat = torch.cat((self.y_hat, y_hat), dim=0)[-200:]   
        
        tp_batch, fp_batch, fn_batch = 0, 0, 0
        n = len(self.y_true)
        y_true = torch.tensor(self.y_true)
        y_hat = torch.tensor(self.y_hat)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip pairs where one or both points are noise
                if y_hat[i]==-1 or y_hat[j]==-1:
                    continue

                same_cluster = y_hat[i] == y_hat[j]
                same_class = y_true[i] == y_true[j]
                if same_cluster and same_class:
                    tp_batch += 1 
                elif same_cluster and not same_class:
                    fp_batch += 1 
                elif not same_cluster and same_class:
                    fn_batch += 1 

        

        # Calculate step Precision and Recall
        precision = tp_batch / (tp_batch + fp_batch) if (tp_batch + fp_batch) > 0 else 0
        recall = tp_batch / (tp_batch + fn_batch) if (tp_batch + fn_batch) > 0 else 0
        noise_ratio = (y_hat == -1).sum().item()/ y_true.numel() if self.total_samples > 0 else 0
        
        self.latest_precision = torch.tensor(precision)
        self.latest_recall = torch.tensor(recall)
        self.latest_noise_ratio = torch.tensor(noise_ratio)
        
        self.sum_precision += self.latest_precision
        self.sum_recall += self.latest_recall
        self.step_counter += 1

    def compute(self):
        precision = self.sum_precision / self.step_counter
        recall = self.sum_recall / self.step_counter
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        noise_ratio = self.noise_count / self.total_samples if self.total_samples > 0 else 0

        return {
            "Precision": precision.item(),
            "Recall": recall.item(),
            "F1-Score": f1_score.item(),
            "Noise Ratio": noise_ratio.item()
        }
