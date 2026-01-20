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
        x: (B, T, C) visual token embeddings (i.e. from VisualEncoder)
        Returns:
            x_masked: masked embeddings
            mask: binary mask (1 = masked, 0 = kept)
            loss_gauss, loss_kl, loss_div: mask regularization losses
        """
        # Train in 2 separate phases:
        #   - AVFF frozen & train module with soft masking
        #   - AVFF unfrozen & train with hard masking module

        B, T, _ = x.shape # batch, length, dim
        len_keep = int(T * (1 - mask_ratio))

        # Predict soft mask scores from MaskingNet
        mask_embedding = self.masking_net(x)  # (B, L): sigmoid [0, 1]

        if not hard_mask:
            x_masked = x * (1 - mask_embedding.unsqueeze(-1))

            return x_masked, mask_embedding, None
        else:
            # Compute hard binary mask by top-k thresholding
            ids_shuffle = torch.argsort(mask_embedding, dim=1, descending=True) # descend: small is remove, large is keep
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]

            # Binary mask (1 = drop, 0 = keep)
            mask = torch.ones([B, T], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # Apply mask (hard masking)
            x_masked = x.clone()
            x_masked[mask.bool()] = 0.0  # zero out masked tokens

            return x_masked, mask, ids_restore


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
