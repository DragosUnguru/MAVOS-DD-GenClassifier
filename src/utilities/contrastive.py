"""
Contrastive loss functions for domain-invariant feature learning.

The goal is to train the masking network to produce features that are INVARIANT
to the generative method used (domain confusion), which helps generalization to
unseen generative methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Can be used in two modes:
    1. Standard supervised contrastive: Pull together same-class samples, push apart different-class
    2. Domain confusion (reverse): Push apart same-domain samples, pull together different-domain
    
    Reference: https://arxiv.org/abs/2004.11362 (Supervised Contrastive Learning)
    
    Args:
        temperature: Softmax temperature for scaling similarities
        contrast_mode: 'all' uses all samples as anchors, 'one' uses first sample only
        base_temperature: Base temperature for normalization
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for a batch of features.
        
        Args:
            features: Hidden vectors of shape (batch_size, feature_dim) or
                      (batch_size, n_views, feature_dim) if using multiple augmented views
            labels: Ground truth labels of shape (batch_size,). Can be:
                    - Integer class labels for single-label classification
                    - Multi-hot vectors (batch_size, n_classes) for multi-label
            mask: Contrastive mask of shape (batch_size, batch_size).
                  mask[i,j] = 1 means i and j are positive pairs.
                  If None, constructed from labels.
        
        Returns:
            A scalar loss value
        """
        device = features.device
        
        # Handle different input shapes
        if len(features.shape) < 3:
            # Add view dimension: (B, D) -> (B, 1, D)
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot specify both labels and mask')
        
        if mask is None and labels is None:
            # Self-supervised mode: each sample is only positive with its augmented views
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous()
            
            # Handle multi-hot labels (e.g., gen_labels with multiple active classes)
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # Multi-hot: compute similarity as dot product of label vectors
                # Samples are positive if they share at least one label
                labels_norm = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
                mask = torch.mm(labels, labels.t())  # Overlap count
                mask = (mask > 0).float()  # Binary: any shared label
            else:
                # Single label: standard equality check
                if len(labels.shape) > 1:
                    labels = labels.squeeze()
                mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        # Expand mask for n_views
        # (B, B) -> (B*n_views, B*n_views)
        mask = mask.repeat(n_views, n_views)
        
        # Flatten features: (B, n_views, D) -> (B*n_views, D)
        contrast_features = features.view(batch_size * n_views, -1)
        
        # Normalize features
        contrast_features = F.normalize(contrast_features, dim=1)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_features, contrast_features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=device)
        mask = mask * logits_mask
        
        # Compute log softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero for samples with no positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()
        
        return loss


class DomainConfusionLoss(nn.Module):
    """
    Domain Confusion Loss for making features invariant to domain (generative method).
    
    This is the OPPOSITE of supervised contrastive: we want features from DIFFERENT
    domains to be SIMILAR, and features from the SAME domain to be DIFFERENT.
    
    The idea is that if the masking network learns to produce features that look
    similar regardless of which generative method was used, it will generalize better
    to unseen methods.
    
    Implementation: We use negative supervised contrastive loss, OR we can maximize
    entropy of domain predictions.
    
    Args:
        temperature: Softmax temperature
        mode: 'contrastive' - reverse supervised contrastive
              'entropy' - maximize entropy of domain predictions
    """
    
    def __init__(self, temperature=0.07, mode='contrastive'):
        super(DomainConfusionLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        
        if mode == 'contrastive':
            self.sup_con = SupConLoss(temperature=temperature)
    
    def forward(self, features, domain_labels=None, domain_logits=None):
        """
        Compute domain confusion loss.
        
        Args:
            features: Feature vectors (batch_size, feature_dim)
            domain_labels: Domain (generative method) labels
                           Multi-hot: (batch_size, n_domains)
                           Or integer: (batch_size,)
            domain_logits: If mode='entropy', logits from domain classifier
        
        Returns:
            Scalar loss value
        """
        if self.mode == 'contrastive':
            if domain_labels is None:
                raise ValueError("domain_labels required for contrastive mode")
            
            # REVERSE the mask: samples are positive if they have DIFFERENT domains
            # This is achieved by using (1 - mask) in the SupConLoss
            device = features.device
            batch_size = features.shape[0]
            
            # Compute standard positive mask
            if len(domain_labels.shape) > 1 and domain_labels.shape[1] > 1:
                # Multi-hot labels
                same_domain_mask = torch.mm(domain_labels, domain_labels.t())
                same_domain_mask = (same_domain_mask > 0).float()
            else:
                if len(domain_labels.shape) > 1:
                    domain_labels = domain_labels.squeeze()
                same_domain_mask = torch.eq(
                    domain_labels.unsqueeze(0), 
                    domain_labels.unsqueeze(1)
                ).float()
            
            # Invert: different domain = positive pair for domain confusion
            different_domain_mask = 1 - same_domain_mask
            
            # Remove self-pairs from positive set
            eye = torch.eye(batch_size, device=device)
            different_domain_mask = different_domain_mask * (1 - eye)
            
            # If all samples are same domain, no loss
            if different_domain_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Use SupConLoss with inverted mask
            return self.sup_con(features, mask=different_domain_mask)
        
        elif self.mode == 'entropy':
            if domain_logits is None:
                raise ValueError("domain_logits required for entropy mode")
            
            # Maximize entropy of predictions = uniform distribution over domains
            probs = F.softmax(domain_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # We want to MAXIMIZE entropy, so return negative
            return -entropy.mean()
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Also known as InfoNCE loss.
    
    Used for self-supervised contrastive learning where augmented views
    of the same sample are positive pairs.
    
    Args:
        temperature: Temperature scaling factor
    """
    
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for two views.
        
        Args:
            z_i: First view embeddings (batch_size, feature_dim)
            z_j: Second view embeddings (batch_size, feature_dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate to get all representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, feature_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # Positive pair indices
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to a lower-dimensional space where contrastive loss is computed.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension (contrastive space)
        n_layers: Number of MLP layers (2 or 3)
    """
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=128, n_layers=2):
        super(ProjectionHead, self).__init__()
        
        layers = []
        
        if n_layers == 2:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ])
        elif n_layers == 3:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ])
        else:
            raise ValueError(f"n_layers must be 2 or 3, got {n_layers}")
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.projector(x)
