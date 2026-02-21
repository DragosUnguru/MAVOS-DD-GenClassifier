import math
import torch
import torch.nn as nn
from .audio_modules import AudioEncoder, AudioDecoder
from .visual_modules import VisualEncoder, VisualDecoder
from .fusion_modules import A2VNetwork, V2ANetwork
from .masking_module.masking_modules import MaskingNet
import random
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_rate=[0.5, 0.5]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop1 = nn.Dropout(p=drop_rate[0])
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=drop_rate[1])
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.act2(x)
        x = self.fc3(x)
        
        return x

class VideoCAVMAE(nn.Module):
    def __init__(self, 
        img_size=224,
        patch_size=16, 
        n_frames=16, 
        audio_length=1024,
        mel_bins=128,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        norm_pix_loss=True,
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.n_frames = n_frames
        
        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.audio_decoder = AudioDecoder(
            num_patches=audio_length * mel_bins // (patch_size ** 2),
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.visual_encoder = VisualEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            n_frames=n_frames, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.visual_decoder = VisualDecoder(
            img_size=img_size, 
            patch_size=patch_size, 
            n_frames=n_frames, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.a2v = A2VNetwork(
            audio_dim=64 * self.n_frames // 4,
            visual_dim=196 * self.n_frames // 4,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
        self.v2a = V2ANetwork(
            audio_dim=64 * self.n_frames // 4,
            visual_dim=196 * self.n_frames // 4,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
    
    def forward_mse_loss(self, target, pred):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        
        loss = (pred - target).pow(2)
        loss = loss.mean()
    
        return loss

    
    def forward_contrastive(self, audio_rep, video_rep, bidrectional_contrast=True):
        # calculate nce loss for mean-visual representation and mean-audio representation
        
        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
        
        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05
        
        if not bidrectional_contrast:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc
    
    def complementary_mask(self, audio_emb, video_emb):
        # Determine the batch size and dimensions
        batch_size, audio_seq_len, _, audio_feat_dim = audio_emb.shape
        _, video_seq_len, _, video_feat_dim = video_emb.shape

        # Initialize masks for audio and video with the same size as their embeddings
        audio_mask = torch.ones((audio_emb.shape[0], audio_emb.shape[1]), dtype=torch.bool).to(self.device)
        video_mask = torch.zeros((video_emb.shape[0], video_emb.shape[1]), dtype=torch.bool).to(self.device)

        # Define the number of masks to apply (using the minimum sequence length)
        num_masks = min(audio_seq_len, video_seq_len)

        # Apply masks
        idxs = random.sample([i for i in range(num_masks)], 4)
        for idx in idxs:
            audio_mask[:, idx] = 0
            video_mask[:, idx] = 1

        # Expand masks to match the feature dimensions
        audio_mask = audio_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, audio_emb.shape[2], audio_emb.shape[3])
        video_mask = video_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, video_emb.shape[2], video_emb.shape[3])

        # Apply the masks to the embeddings
        audio_masked = audio_emb * audio_mask.float()
        video_masked = video_emb * video_mask.float()
        
        # Return only the visible patch
        b, _, t, c = audio_emb.shape
        audio_visible = audio_masked[audio_mask].reshape(b, -1, t, c)
        b, _, t, c = video_emb.shape
        video_visible = video_masked[video_mask].reshape(b, -1, t, c)

        return audio_visible, video_visible, audio_mask, video_mask
    
    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, audio, video, mae_loss_weight=1.0, contrast_loss_weight=0.01):
        # audio: (B, 1024, 128)
        # video: (B, 3, 16, 224, 224)
        
        # Forward audio and video through their respective encoders
        audio_emb = self.audio_encoder(audio)
        video_emb = self.visual_encoder(video)
        
        # Compute comtrastive loss
        nce_loss, c_acc = self.forward_contrastive(audio_emb.mean(dim=1), video_emb.mean(dim=1))
        
        # Rearrange audio and video embeddings to perform temporal complementary mask
        b, t, c = audio_emb.shape
        audio_emb = audio_emb.reshape(b, self.n_frames // 2, -1, c)
        b, t, c = video_emb.shape
        video_emb = video_emb.reshape(b, self.n_frames // 2, -1, c)
        
        # Perform complementary Mask
        audio_visible, video_visible, audio_mask, video_mask = self.complementary_mask(audio_emb, video_emb)
        
        a2v_emb = self.a2v(audio_visible)
        v2a_emb = self.v2a(video_visible)
        
        # Concate the visible patches and a2v/v2a patches
        audio_fusion = torch.zeros_like(audio_emb)
        video_fusion = torch.zeros_like(video_emb)
        audio_fusion[audio_mask] = rearrange(audio_visible, 'b t c d -> (b t c d)')
        audio_fusion[~audio_mask] = rearrange(v2a_emb, 'b t c d -> (b t c d)')
        video_fusion[video_mask] = rearrange(video_visible, 'b t c d -> (b t c d)')
        video_fusion[~video_mask] = rearrange(a2v_emb, 'b t c d -> (b t c d)')
        
        audio_fusion = rearrange(audio_fusion, 'b t c d -> b (t c) d')
        video_fusion = rearrange(video_fusion, 'b t c d -> b (t c) d')
        audio_recon = self.audio_decoder(audio_fusion)
        video_recon = self.visual_decoder(video_fusion)
        
        audio_input = audio.unsqueeze(1)
        audio_input = audio_input.transpose(2, 3)
        audio_recon = self.unpatchify(audio_recon, 1, audio_input.shape[2]//16, audio_input.shape[3]//16, 16)
        video_recon = self.visual_decoder.unpatch_to_img(video_recon)
        
        # Compute Reconstruction loss
        rec_loss_a = self.forward_mse_loss(audio_input, audio_recon)
        rec_loss_v = self.forward_mse_loss(video, video_recon)
        
        #Compute total loss
        total_loss = mae_loss_weight * (rec_loss_v + rec_loss_a) + contrast_loss_weight * nce_loss
        
        return total_loss, nce_loss, c_acc, rec_loss_a, rec_loss_v, audio_recon, video_recon
    

class DiversityLoss(nn.Module):
    def __init__(self, distance_metric='euclidean', scale_factor=0.1):
        super(DiversityLoss, self).__init__()
        self.distance_metric = distance_metric
        self.scale_factor = scale_factor

    def forward(self, outputs):
        n = outputs.size()[0]  # size of batch
        if n == 1:
            return 0.0
        # compute distance matrix between outputs
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(outputs, outputs, p=2)
        else:
            raise ValueError('Unsupported distance metric')
        # compute diversity loss
        loss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                loss += torch.exp(-distances[i, j] * self.scale_factor)
        # normalize loss by batch size
        loss /= n * (n - 1) / 2
        return loss


class VideoCAVMAEFT(nn.Module):
    def __init__(self, 
        n_classes=2,
        n_gen_classes=9,  # Number of generative method classes (4 video + 5 audio)
        img_size=224,
        patch_size=16, 
        n_frames=16, 
        audio_length=1024,
        mel_bins=128,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        norm_pix_loss=True,
        lambda_gauss=1.0,
        lambda_kl=0.01,
        lambda_diversity=2.0,
        lambda_adv=1.0,  # Adversarial loss weight
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.n_frames = n_frames
        self.n_gen_classes = n_gen_classes

        # MAE masking net specifics
        self.alpha = -1 / (0.12 * 0.12 * 2)
        self.beta = 1 / (0.12 * math.sqrt(2 * math.pi))
        self.diversity_loss = DiversityLoss()

        self.lambda_gauss = lambda_gauss
        self.lambda_kl = lambda_kl
        self.lambda_diversity = lambda_diversity
        self.lambda_adv = lambda_adv

        self.masking_net = MaskingNet(
            num_tokens=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio
            # norm_layer=nn.LayerNorm
        )
        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.visual_encoder = VisualEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            n_frames=n_frames, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.a2v = A2VNetwork(
            audio_dim=64 * self.n_frames // 2,
            visual_dim=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
        self.v2a = V2ANetwork(
            audio_dim=64 * self.n_frames // 2,
            visual_dim=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
        
        hidden_dim = 1024
        self.mlp_vision = torch.nn.Linear(1568, hidden_dim)
        self.mlp_audio = torch.nn.Linear(512, hidden_dim)
        self.mlp_head = MLP(input_size=hidden_dim * 2, hidden_size=hidden_dim, num_classes=n_classes)
        
        # Adversarial head for generative method classification
        self.gen_classifier = MLP(
            input_size=hidden_dim * 2, 
            hidden_size=hidden_dim, 
            num_classes=n_gen_classes,
            drop_rate=[0.5, 0.5]
        )

    def kl_divergence(self, p, q):
        p = torch.as_tensor(p, dtype=torch.float32)
        q = torch.as_tensor(q, dtype=torch.float32)
        return torch.sum(p * torch.log(p / q), dim=-1)

    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def apply_masking(self, x, hard_mask, mask_ratio):
        """
        Apply learned masking to visual token embeddings.
        
        x: (B, T, C) visual token embeddings (i.e. from VisualEncoder)
        
        For hard_mask=True, uses straight-through estimator to maintain gradient flow:
        - Forward pass: uses hard binary mask (0 or 1)
        - Backward pass: gradients flow through soft mask_embedding
        
        Returns:
            x_masked: masked embeddings
            mask: binary mask (1 = masked, 0 = kept) or soft mask
            ids_restore: indices to restore original order (for hard_mask=True)
        """
        B, T, _ = x.shape  # batch, length, dim
        len_keep = int(T * (1 - mask_ratio))

        # Predict soft mask scores from MaskingNet (values in [0, 1] after sigmoid)
        mask_embedding = self.masking_net(x)  # (B, T)

        if not hard_mask:
            # Soft masking: differentiable, gradients flow naturally
            x_masked = x * (1 - mask_embedding.unsqueeze(-1))
            return x_masked, mask_embedding, None
        else:
            # Hard masking with straight-through estimator for gradient flow
            
            # Sort to find top-k tokens to keep (descending: higher score = more important = keep)
            ids_shuffle = torch.argsort(mask_embedding, dim=1, descending=True)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Create hard binary mask (1 = drop, 0 = keep)
            hard_mask_binary = torch.ones([B, T], device=x.device)
            hard_mask_binary[:, :len_keep] = 0
            hard_mask_binary = torch.gather(hard_mask_binary, dim=1, index=ids_restore)

            # Straight-through estimator:
            # Forward: use hard_mask_binary (discrete 0/1)
            # Backward: use mask_embedding (continuous, differentiable)
            # The trick: hard_mask_binary - mask_embedding.detach() + mask_embedding
            # This equals hard_mask_binary in forward, but has gradient of mask_embedding in backward
            soft_mask = mask_embedding  # continuous [0, 1]
            straight_through_mask = hard_mask_binary - soft_mask.detach() + soft_mask

            # Apply mask: multiply by (1 - mask) to zero out masked tokens
            x_masked = x * (1 - straight_through_mask).unsqueeze(-1)

            return x_masked, hard_mask_binary, ids_restore


    def forward(self, audio, video, apply_mask=False, hard_mask=False, hard_mask_ratio=0.75, adversarial=False, detach_features_for_gen=False):
        # audio: (B, 1024, 128)
        # video: (B, 3, 16, 224, 224)

        # Forward audio and video through their respective encoders
        audio_emb = self.audio_encoder(audio)
        video_emb = self.visual_encoder(video)
        video_mask = None
        ids_restore = None

        # Apply learned masking on visual embeddings
        if apply_mask:
            video_emb, video_mask, ids_restore = self.apply_masking(video_emb, hard_mask, hard_mask_ratio)

        # Rearrange audio and video embeddings to perform temporal complementary mask
        b, t, c = audio_emb.shape
        audio_emb = audio_emb.reshape(b, self.n_frames // 2, -1, c)
        b, t, c = video_emb.shape
        video_emb = video_emb.reshape(b, self.n_frames // 2, -1, c)

        video_fusion = self.a2v(audio_emb)
        audio_fusion = self.v2a(video_emb)

        # Concat along feature dimension
        video_fusion = torch.concat((video_fusion, video_emb), dim=-1)
        audio_fusion = torch.concat((audio_fusion, audio_emb), dim=-1)
        video_fusion = video_fusion.mean(dim=-1)
        audio_fusion = audio_fusion.mean(dim=-1)
        
        video_fusion = rearrange(video_fusion, 'b t c -> b (t c)')
        audio_fusion = rearrange(audio_fusion, 'b t c -> b (t c)')
        
        video_fusion = self.mlp_vision(video_fusion)
        audio_fusion = self.mlp_audio(audio_fusion)
        
        fused_features = torch.concat((video_fusion, audio_fusion), dim=-1)
        
        # Main classification head (real/fake detection)
        output = self.mlp_head(fused_features)
        
        # Generative method classification head
        gen_output = None
        if adversarial:
            if detach_features_for_gen:
                # Detach features: only gen_classifier receives gradients
                # Used in Step 1 (discriminator update)
                gen_output = self.gen_classifier(fused_features.detach())
            else:
                # Normal forward: gradients flow through everything
                # Used in Step 2 (generator update) - masking_net learns to fool gen_classifier
                gen_output = self.gen_classifier(fused_features)
        
        return output, gen_output, video_mask, ids_restore
    
    def set_adversarial_lambda(self, lambda_val):
        """Set the gradient reversal strength for adversarial training."""
        self.gradient_reversal.set_lambda(lambda_val)


    def freeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = False


    def unfreeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = True


    def freeze_backbone(self):
        """Freeze everything except masking_net, gen_classifier, and pos_embed."""
        for name, param in self.named_parameters():
            if "masking_net" in name:
                continue
            if "pos_embed" in name:
                continue
            if "gen_classifier" in name:
                continue

            param.requires_grad = False


    def unfreeze_backbone(self):
        """Unfreeze everything except masking_net and pos_embed."""
        for name, param in self.named_parameters():
            if "masking_net" in name:
                continue
            if "pos_embed" in name:
                continue
            param.requires_grad = True


    def freeze_gen_classifier(self):
        """Freeze the generative method classifier head."""
        for name, param in self.named_parameters():
            if "gen_classifier" in name:
                param.requires_grad = False


    def unfreeze_gen_classifier(self):
        """Unfreeze the generative method classifier head."""
        for name, param in self.named_parameters():
            if "gen_classifier" in name:
                param.requires_grad = True


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps embeddings to a lower-dimensional space where contrastive loss is computed.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class VideoCAVMAEContrastive(nn.Module):
    """
    Video-Audio Contrastive Learning Model for Deepfake Detection.
    
    Key idea: Learn representations by contrasting:
    1. Real vs Fake samples (supervised contrastive)
    2. Same-class samples should be close, different-class samples should be far
    3. Audio-video pairs from the same sample should be closer than mismatched pairs
    
    This approach learns more generalizable features compared to adversarial training
    by explicitly structuring the embedding space based on sample relationships.
    """
    def __init__(self, 
        n_classes=2,
        img_size=224,
        patch_size=16, 
        n_frames=16, 
        audio_length=1024,
        mel_bins=128,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        norm_pix_loss=True,
        projection_dim=128,  # Dimension of contrastive embedding space
        temperature=0.07,    # Temperature for InfoNCE loss
    ):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.temperature = temperature
        self.projection_dim = projection_dim

        # Masking net for learned attention-based masking (optional, can be disabled)
        self.masking_net = MaskingNet(
            num_tokens=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio
        )
        
        self.audio_encoder = AudioEncoder(
            audio_length=audio_length,
            mel_bins=mel_bins,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            encoder_depth=encoder_depth,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.visual_encoder = VisualEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            n_frames=n_frames, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.a2v = A2VNetwork(
            audio_dim=64 * self.n_frames // 2,
            visual_dim=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
        self.v2a = V2ANetwork(
            audio_dim=64 * self.n_frames // 2,
            visual_dim=196 * self.n_frames // 2,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads
        )
        
        hidden_dim = 1024
        self.mlp_vision = torch.nn.Linear(1568, hidden_dim)
        self.mlp_audio = torch.nn.Linear(512, hidden_dim)
        
        # Classification head (real/fake detection)
        self.mlp_head = MLP(input_size=hidden_dim * 2, hidden_size=hidden_dim, num_classes=n_classes)
        
        # Projection head for real/fake contrastive clustering
        self.fusion_projector = ProjectionHead(hidden_dim * 2, hidden_dim, projection_dim)
        
        # Projector for generative method discrimination (adversarial objective)
        # This projector learns to cluster samples by their generative method
        # The masking_net will try to FOOL this projector (adversarial)
        self.gen_method_projector = ProjectionHead(hidden_dim * 2, hidden_dim, projection_dim)
        
        # MAE masking net specifics (for mask regularization if used)
        self.alpha = -1 / (0.12 * 0.12 * 2)
        self.beta = 1 / (0.12 * math.sqrt(2 * math.pi))
        self.diversity_loss = DiversityLoss()
        self.lambda_gauss = 1.0
        self.lambda_kl = 0.01
        self.lambda_diversity = 2.0

    def kl_divergence(self, p, q):
        p = torch.as_tensor(p, dtype=torch.float32)
        q = torch.as_tensor(q, dtype=torch.float32)
        return torch.sum(p * torch.log(p / q), dim=-1)

    def apply_masking(self, x, hard_mask, mask_ratio):
        """
        Apply learned masking to visual tokens.
        
        For hard_mask=True, uses straight-through estimator to maintain gradient flow:
        - Forward pass: uses hard binary mask (0 or 1)
        - Backward pass: gradients flow through soft mask_embedding
        
        This allows the masking_net to learn which tokens to mask while still
        performing discrete masking during inference.
        """
        B, T, _ = x.shape
        len_keep = int(T * (1 - mask_ratio))
        
        # Predict soft mask scores from MaskingNet (values in [0, 1] after sigmoid)
        mask_embedding = self.masking_net(x)  # (B, T)
        
        if not hard_mask:
            # Soft masking: differentiable, gradients flow naturally
            x_masked = x * (1 - mask_embedding.unsqueeze(-1))
            return x_masked, mask_embedding, None
        else:
            # Hard masking with straight-through estimator for gradient flow
            
            # Sort to find top-k tokens to keep (higher score = more important = keep)
            ids_shuffle = torch.argsort(mask_embedding, dim=1, descending=True)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # Create hard binary mask (1 = drop, 0 = keep)
            hard_mask_binary = torch.ones([B, T], device=x.device)
            hard_mask_binary[:, :len_keep] = 0
            hard_mask_binary = torch.gather(hard_mask_binary, dim=1, index=ids_restore)
            
            # Straight-through estimator:
            # Forward: use hard_mask_binary (discrete 0/1)
            # Backward: use mask_embedding (continuous, differentiable)
            # This equals hard_mask_binary in forward, but has gradient of mask_embedding in backward
            soft_mask = mask_embedding  # continuous [0, 1]
            straight_through_mask = hard_mask_binary - soft_mask.detach() + soft_mask
            
            # Apply mask: multiply by (1 - mask) to zero out masked tokens
            x_masked = x * (1 - straight_through_mask).unsqueeze(-1)
            
            return x_masked, hard_mask_binary, ids_restore

    def supervised_contrastive_loss(self, features, labels, temperature=None):
        """
        Supervised Contrastive Loss (SupCon).
        
        Pulls together samples with the same label while pushing apart samples
        with different labels in the embedding space.
        
        Args:
            features: (B, D) normalized feature embeddings
            labels: (B,) or (B, C) class labels (will be converted to class indices)
            temperature: scaling temperature (default: self.temperature)
        
        Returns:
            loss: scalar contrastive loss
        """
        if temperature is None:
            temperature = self.temperature
        
        device = features.device
        batch_size = features.shape[0]
        
        # Handle one-hot encoded labels
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)
        
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast (diagonal)
        mask_self = torch.eye(batch_size, device=device)
        mask_positives = mask_positives - mask_self
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * (1 - mask_self)  # Exclude self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute mean of log-likelihood over positive pairs
        num_positives = mask_positives.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)  # Avoid division by zero
        
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / num_positives
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss

    def cross_modal_contrastive_loss(self, audio_features, video_features, temperature=None):
        """
        Cross-modal contrastive loss (similar to CLIP).
        
        Encourages audio and video embeddings from the same sample to be similar,
        while embeddings from different samples should be dissimilar.
        
        Args:
            audio_features: (B, D) audio embeddings
            video_features: (B, D) video embeddings
            temperature: scaling temperature
        
        Returns:
            loss: scalar contrastive loss
            accuracy: matching accuracy
        """
        if temperature is None:
            temperature = self.temperature
        
        # Normalize features
        audio_features = torch.nn.functional.normalize(audio_features, dim=1)
        video_features = torch.nn.functional.normalize(video_features, dim=1)
        
        batch_size = audio_features.shape[0]
        
        # Compute similarity matrix
        logits = torch.matmul(audio_features, video_features.T) / temperature
        
        # Labels are diagonal (audio[i] should match video[i])
        labels = torch.arange(batch_size, device=audio_features.device)
        
        # Symmetric loss (audio->video and video->audio)
        loss_a2v = torch.nn.functional.cross_entropy(logits, labels)
        loss_v2a = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_a2v + loss_v2a) / 2
        
        # Compute accuracy
        with torch.no_grad():
            pred_a2v = logits.argmax(dim=1)
            pred_v2a = logits.T.argmax(dim=1)
            acc = ((pred_a2v == labels).float().mean() + (pred_v2a == labels).float().mean()) / 2
        
        return loss, acc

    def forward(self, audio, video, apply_mask=False, hard_mask=False, hard_mask_ratio=0.75, 
                return_projections=False):
        """
        Forward pass with optional contrastive projections.
        
        Args:
            audio: (B, 1024, 128) mel spectrogram
            video: (B, 3, 16, 224, 224) video frames
            apply_mask: whether to apply learned masking
            hard_mask: use hard (binary) vs soft masking
            hard_mask_ratio: ratio of tokens to mask
            return_projections: return projected features for contrastive loss
        
        Returns:
            output: (B, n_classes) classification logits
            video_mask: mask tensor (if apply_mask=True)
            projections: dict of projected features (if return_projections=True)
        """
        # Encode audio and video
        audio_emb = self.audio_encoder(audio)
        video_emb = self.visual_encoder(video)
        video_mask = None
        ids_restore = None

        # Apply learned masking on visual embeddings (optional)
        if apply_mask:
            video_emb, video_mask, ids_restore = self.apply_masking(video_emb, hard_mask, hard_mask_ratio)

        # Rearrange for temporal fusion
        b, t, c = audio_emb.shape
        audio_emb = audio_emb.reshape(b, self.n_frames // 2, -1, c)
        b, t, c = video_emb.shape
        video_emb = video_emb.reshape(b, self.n_frames // 2, -1, c)

        # Cross-modal fusion
        video_fusion = self.a2v(audio_emb)
        audio_fusion = self.v2a(video_emb)

        # Concat along feature dimension
        video_fusion = torch.concat((video_fusion, video_emb), dim=-1)
        audio_fusion = torch.concat((audio_fusion, audio_emb), dim=-1)
        video_fusion = video_fusion.mean(dim=-1)
        audio_fusion = audio_fusion.mean(dim=-1)
        
        video_fusion = rearrange(video_fusion, 'b t c -> b (t c)')
        audio_fusion = rearrange(audio_fusion, 'b t c -> b (t c)')
        
        # Project to hidden space
        video_features = self.mlp_vision(video_fusion)
        audio_features = self.mlp_audio(audio_fusion)
        
        # Fused features for classification
        fused_features = torch.concat((video_features, audio_features), dim=-1)
        
        # Classification output
        output = self.mlp_head(fused_features)
        
        # Compute projections for contrastive learning
        projections = None
        if return_projections:
            projections = {
                'fusion': self.fusion_projector(fused_features),
                'fused_features': fused_features,  # Raw features for gen_method_projector
            }
        
        return output, video_mask, projections

    def compute_contrastive_loss(self, projections, labels):
        """
        Compute supervised contrastive loss for real/fake clustering.
        
        This is the main contrastive objective: cluster real samples together
        and fake samples together, regardless of generative method.
        
        Args:
            projections: dict from forward() with return_projections=True
            labels: (B, 2) one-hot [is_fake, is_real] or (B,) binary labels
        
        Returns:
            supcon_loss: supervised contrastive loss
        """
        supcon_loss = self.supervised_contrastive_loss(
            projections['fusion'], labels
        )
        return supcon_loss
    
    def compute_method_discrimination_loss(self, projections, gen_labels):
        """
        Compute the method discrimination loss for adversarial training.
        
        This is the DISCRIMINATOR objective: cluster samples by their generative method.
        
        Args:
            projections: dict from forward() with return_projections=True
                         Must contain 'fused_features' key
            gen_labels: (B, n_methods) multi-hot generative method labels
        
        Returns:
            loss: method contrastive loss
            method_acc: accuracy of method clustering
        """
        # Project fused features through gen_method_projector
        gen_method_proj = self.gen_method_projector(projections['fused_features'])
        
        # Compute multi-label method contrastive loss
        loss, method_acc = self.method_contrastive_loss(gen_method_proj, gen_labels)
        
        return loss, method_acc
    
    def compute_adversarial_method_loss(self, projections, gen_labels):
        """
        Compute the adversarial method loss for the generator (masking_net).
        
        This is the GENERATOR objective: make samples INDISTINGUISHABLE by method.
        We want to MAXIMIZE the method contrastive loss (fool the discriminator).
        
        Args:
            projections: dict from forward() with return_projections=True
            gen_labels: (B, n_methods) multi-hot generative method labels
        
        Returns:
            adv_loss: negative of method contrastive loss (to maximize it)
            method_acc: accuracy of method clustering (for monitoring)
        """
        # Project fused features through gen_method_projector
        gen_method_proj = self.gen_method_projector(projections['fused_features'])
        
        # Compute multi-label method contrastive loss
        method_loss, method_acc = self.method_contrastive_loss(gen_method_proj, gen_labels)
        
        # Adversarial: MAXIMIZE the method loss = MINIMIZE negative of it
        adv_loss = -method_loss
        
        return adv_loss, method_acc

    def freeze_backbone(self):
        """Freeze encoder backbone, keep projectors and classifier trainable."""
        for name, param in self.named_parameters():
            if any(x in name for x in ['masking_net', 'pos_embed', 'projector', 'mlp_head']):
                continue
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze encoder backbone."""
        for name, param in self.named_parameters():
            if any(x in name for x in ['masking_net', 'pos_embed']):
                continue
            param.requires_grad = True

    def freeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = False

    def unfreeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = True

    def freeze_gen_method_projector(self):
        """Freeze the generative method projector (discriminator in adversarial setup)."""
        for name, param in self.named_parameters():
            if "gen_method_projector" in name:
                param.requires_grad = False

    def unfreeze_gen_method_projector(self):
        """Unfreeze the generative method projector."""
        for name, param in self.named_parameters():
            if "gen_method_projector" in name:
                param.requires_grad = True

    def method_contrastive_loss(self, features, method_labels, temperature=None):
        """
        Multi-Label Supervised Contrastive Loss for generative method clustering.
        
        This loss handles MULTI-LABEL scenarios where each sample can have multiple
        generative methods (e.g., 1 audio method + 1 video method = up to 2 labels).
        
        Two samples are considered "positive pairs" if they share ANY method in common.
        The similarity weight is proportional to how many methods they share.
        
        Used as the discriminator objective in adversarial training:
        - Discriminator wants to MINIMIZE this loss (cluster by shared methods)
        - Masking net wants to MAXIMIZE this loss (make methods indistinguishable)
        
        Args:
            features: (B, D) feature embeddings
            method_labels: (B, n_methods) multi-hot labels (can have multiple 1s per sample)
            temperature: scaling temperature
        
        Returns:
            loss: scalar contrastive loss
            method_acc: accuracy of method clustering (for monitoring)
        """
        if temperature is None:
            temperature = self.temperature
        
        device = features.device
        batch_size = features.shape[0]
        
        # Ensure method_labels is multi-hot format (B, n_methods)
        if method_labels.dim() == 1:
            raise ValueError("method_labels should be multi-hot (B, n_methods), not class indices")
        
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Compute label similarity: how many methods do samples share?
        # method_labels: (B, n_methods), each row is multi-hot
        # label_similarity[i,j] = number of shared methods between sample i and j
        label_similarity = torch.matmul(method_labels, method_labels.T)  # (B, B)
        
        # Create positive mask: samples share at least one method
        # Weight by number of shared methods (normalized)
        mask_positives = (label_similarity > 0).float()
        
        # Remove self-contrast (diagonal)
        mask_self = torch.eye(batch_size, device=device)
        mask_positives = mask_positives * (1 - mask_self)
        
        # Optional: weight positives by number of shared labels
        # This gives higher weight to samples that share MORE methods
        # Normalize by max possible shared methods (2 in this case)
        positive_weights = label_similarity * (1 - mask_self)
        positive_weights = positive_weights / (positive_weights.max() + 1e-6)
        positive_weights = torch.clamp(positive_weights, min=0)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * (1 - mask_self)  # Exclude self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute weighted mean of log-likelihood over positive pairs
        num_positives = mask_positives.sum(dim=1)
        has_positives = num_positives > 0
        num_positives = torch.clamp(num_positives, min=1)
        
        # Use positive_weights instead of binary mask for weighted loss
        weighted_log_prob_pos = (positive_weights * log_prob).sum(dim=1)
        mean_log_prob_pos = weighted_log_prob_pos / num_positives
        
        # Only compute loss for samples that have positive pairs
        if has_positives.sum() > 0:
            loss = -mean_log_prob_pos[has_positives].mean()
        else:
            # Return a zero loss that's still connected to the computation graph
            # by multiplying with a sum of the features (maintains gradient flow)
            loss = (features * 0).sum()
        
        # Compute method clustering accuracy (for monitoring)
        # Check if the most similar sample shares at least one method
        with torch.no_grad():
            similarity_matrix_masked = similarity_matrix.clone().float()  # Convert to float32 for stability
            similarity_matrix_masked.fill_diagonal_(float('-inf'))  # Exclude self
            most_similar = similarity_matrix_masked.argmax(dim=1)
            
            # Check if most similar sample shares any method
            shares_method = label_similarity[torch.arange(batch_size, device=device), most_similar] > 0
            method_acc = shares_method.float().mean()
        
        return loss, method_acc

    def multilabel_real_fake_contrastive_loss(self, features, main_labels, method_labels, temperature=None):
        """
        Supervised Contrastive Loss that clusters by real/fake while being
        METHOD-AGNOSTIC within the fake class.
        
        This encourages:
        - Real samples to cluster together
        - Fake samples to cluster together REGARDLESS of their generative method
        
        This is the OPPOSITE of method_contrastive_loss and helps the backbone
        learn method-agnostic fake detection features.
        
        Args:
            features: (B, D) feature embeddings
            main_labels: (B, 2) one-hot [is_fake, is_real] or (B,) binary labels
            method_labels: (B, n_methods) multi-hot labels (not used for clustering, 
                          but can be used for weighting/analysis)
            temperature: scaling temperature
        
        Returns:
            loss: scalar contrastive loss
        """
        if temperature is None:
            temperature = self.temperature
        
        device = features.device
        batch_size = features.shape[0]
        
        # Convert main_labels to binary: 0=fake, 1=real
        if main_labels.dim() > 1:
            # Assuming format [is_fake, is_real]
            is_real = main_labels[:, 1]  # 1 if real, 0 if fake
        else:
            is_real = main_labels
        
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create positive mask: same real/fake class
        is_real = is_real.contiguous().view(-1, 1)
        mask_positives = torch.eq(is_real, is_real.T).float()
        
        # Remove self-contrast
        mask_self = torch.eye(batch_size, device=device)
        mask_positives = mask_positives - mask_self
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute mean of log-likelihood over positive pairs
        num_positives = mask_positives.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)
        
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / num_positives
        
        loss = -mean_log_prob_pos.mean()
        
        return loss