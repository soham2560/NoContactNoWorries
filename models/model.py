from models.AsymFormer import B0_T
from utils.utils import transform

import os
import torch
import torch.nn as nn

# import timm.models.vision_transformer as vit
def generate_causal_mask(seq_len, device=None):
    """
    Returns a (seq_len, seq_len) bool mask
    so that token i cannot attend to tokens j > i.
    Upper-triangular entries are True => masked.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


class CNNDownsampler(nn.Module):
    """
    A small CNN to downsample (120,160,256) -> (new_h, new_w, embed_dim).
    By default, we do two Conv2d layers with stride=2 each time:
      (120->60->30, 160->80->40)
    So final spatial size is (30,40).
    We output 'embed_dim' channels, then we'll flatten for tokens.
    """
    def __init__(self, in_channels: int = 256, embed_dim: int = 256):
        super(CNNDownsampler, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # second downsampling 
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B*H, C=256, H=120, W=160)
        returns: tokens of shape (B*H, new_hw, embed_dim)
                 where new_hw = new_h * new_w, e.g. 40*30 = 1200
        """
        out = self.conv_layers(x) # shape (B*H, embed_dim, new_h=40, new_w=30)
        BHW, embed_dim, new_h, new_w = out.shape
        # Flatten spatial dimensions into tokens
        out = out.view(BHW, embed_dim, new_h * new_w) # shape (B*H, embed_dim, new_hw)
        out = out.permute(0, 2, 1) # shape (B*H, new_hw, embed_dim)

        return out
    
class VisionSelfAttn(nn.Module):
    """
        A single-layer transformer-like block for self-attention among
        the downsampled visual tokens per frame.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        # tokens: (B, N_tokens, embed_dim)
        attn_out, _ = self.mha(tokens, tokens, tokens)  # Self-attention
        out1 = self.ln1(tokens + attn_out)  # Residual + LayerNorm

        ff_out = self.ff(out1)  # Feed-forward
        out2 = self.ln2(out1 + ff_out)  # Residual + LayerNorm

        return out2
    
class ContactTransformer(nn.Module):
    """
        End-to-end example:
        1. Convolutional downsampling per frame -> (new_h) tokens
        2. (optional) self-attention among visual tokens
        3. Pose embedding
        4. Cross-attention: Pose as query
        5. Temporal transformer
        6. Contact head
    """

    def __init__(self,
                in_channels: int = 256,
                embed_dim: int = 256,
                num_heads: int = 4,
                output_dim: int = 70,
                use_vision_attn: bool = True,
                asym_ckpt: str = None,
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        device_id = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

        # 0. Load AsymFormer
        self.asymformer = B0_T()
        if asym_ckpt is not None:
            try:
                checkpoint = torch.load(asym_ckpt, map_location=self.device)
                self.asymformer.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded checkpoint from {asym_ckpt}")
            except FileNotFoundError:
                print(f"Warning: Checkpoint file not found at {asym_ckpt}.")
            except KeyError:
                print(f"Warning: Checkpoint keys do not match model's state_dict.")

        for param in self.asymformer.parameters():
            param.requires_grad = False

        # 1. Convolutional downsampling
        self.downsampler = CNNDownsampler(in_channels, embed_dim)

        # 2. (optional) Self-attention among visual tokens
        self.use_vision_attn = use_vision_attn
        if use_vision_attn:
            self.vision_attn = VisionSelfAttn(embed_dim, num_heads)

        # 3. Pose embedding
        self.pose_emb = nn.Linear(16, embed_dim)

        # 4. Cross-attention: Pose as query
        self.pose_vision_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pose_vision_ln = nn.LayerNorm(embed_dim)

        # 5. Temporal transformer across frames
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 6. Contact head
        self.contact_head = nn.Linear(embed_dim, output_dim)
    
    def forward(self, image, depth, segmask, hand_joint_pose):
        """
            image: (B, Hr, H, W, C)
            depth: (B, Hr, H, W)
            segmask: (B, Hr, H, W)
            hand_joint_pose: (B, Hr, 16)
        """

        B, H, h, w, c = image.shape

        # Apply the segmentation mask to the image and depth, apply transform and get vision features from AsymFormer 

        # image, depth, segmask = image.to(self.device), depth.to(self.device), segmask.to(self.device)
        depth = depth.unsqueeze(-1) # depth: (B, Hr, H, W) -> (B, Hr, H, W, 1)
        segmask = segmask.unsqueeze(-1) # segmask: (B, Hr, H, W) -> (B, Hr, H, W, 1)
        masked_image = (image * segmask).permute(0,1,4,2,3).view(B*H, c, h, w) # masked_image: (B, Hr, H, W, C) -> (B*Hr, C, H, W) -> (B*Hr, C, H, W)
        masked_depth = (depth * segmask).permute(0,1,4,2,3).view(B*H, 1, h, w) # masked_depth: (B, Hr, H, W, 1) -> (B*Hr, 1, H, W) -> (B*Hr, 1, H, W)
        sample = {'image': masked_image, 'depth': masked_depth} # sample: dict with image: (B*Hr, C, H, W), depth: (B*Hr, 1, H, W)
        sample = transform(sample)
        masked_image, masked_depth = sample['image'].to(self.device), sample['depth'].to(self.device) # image: (B*Hr, C, H, W), depth: (B*Hr, 1, H, W)
        vision_features = self.asymformer(masked_image, masked_depth)

        # _,d1,d2,d3 = vision_features.shape
        # vision_features = vision_features.view(B, H, d1, d2, d3) # vision_features: (B, Hr, D, H', W')

        # # Flatten the (B, Hr) dims for CNN downsampling
        # feats_2d = vision_features.view(B*Hr, c, h, w)

        # 1. Convolutional downsampling -> (B*Hr, new_hw, embed_dim)
        conv_tokens = self.downsampler(vision_features) # e.g. shape (B*Hr, 1200, 256)


        # 2. (optional) Self-attention among visual tokens
        if self.use_vision_attn:
            conv_tokens = self.vision_attn(conv_tokens)

        # 3. Pose embedding -> (B, Hr, embed_dim)
        pose_emb = self.pose_emb(hand_joint_pose)

        # Flatten for cross-attention: (B, Hr, embed_dim) -> (B*Hr, 1, embed_dim)
        pose_flat = pose_emb.view(B*H, 1, -1)

        # We want cross-attention:
        # - Query: pose_flat, shape (B*Hr, 1, embed_dim)
        # - Key, Value: conv_tokens, shape (B*Hr, new_hw, embed_dim)
        fused, _ = self.pose_vision_cross_attn(pose_flat, conv_tokens, conv_tokens)
        # fused shape: (B*Hr, 1, embed_dim)

        # Residual + LayerNorm
        fused = self.pose_vision_ln(fused + pose_flat) # shape (B*Hr, 1, embed_dim)

        # Reshape back to (B, Hr, embed_dim)
        fused_frames = fused.view(B, H, -1)

        # 5. CAUSAL temporal self-attention
        #   create a mask: shape (H, H) => each position can't attend future positions
        causal_mask = generate_causal_mask(H, device=fused_frames.device)
        
        # pass mask=causal_mask to TransformerEncoder
        temporal_out = self.temporal_transformer(fused_frames, mask=causal_mask) # shape (B, Hr, embed_dim)
        # temporal_out = self.temporal_transformer(fused_frames) # shape (B, Hr, embed_dim)

        # 6. Contact head
        contact_preds = self.contact_head(temporal_out) # shape (B, Hr, 70)

        return contact_preds
    
class CT_without_segmask(nn.Module):
    """
        End-to-end example:
        1. Convolutional downsampling per frame -> (new_h) tokens
        2. (optional) self-attention among visual tokens
        3. Pose embedding
        4. Cross-attention: Pose as query
        5. Temporal transformer
        6. Contact head
    """

    def __init__(self,
                in_channels: int = 256,
                embed_dim: int = 256,
                num_heads: int = 4,
                output_dim: int = 70,
                use_vision_attn: bool = True,
                asym_ckpt: str = None,
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        device_id = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

        # 0. Load AsymFormer
        self.asymformer = B0_T()
        if asym_ckpt is not None:
            try:
                checkpoint = torch.load(asym_ckpt, map_location=self.device)
                self.asymformer.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded checkpoint from {asym_ckpt}")
            except FileNotFoundError:
                print(f"Warning: Checkpoint file not found at {asym_ckpt}.")
            except KeyError:
                print(f"Warning: Checkpoint keys do not match model's state_dict.")

        for param in self.asymformer.parameters():
            param.requires_grad = False

        # 1. Convolutional downsampling
        self.downsampler = CNNDownsampler(in_channels, embed_dim)

        # 2. (optional) Self-attention among visual tokens
        self.use_vision_attn = use_vision_attn
        if use_vision_attn:
            self.vision_attn = VisionSelfAttn(embed_dim, num_heads)

        # 3. Pose embedding
        self.pose_emb = nn.Linear(16, embed_dim)
        self.desired_pose_emb = nn.Linear(16, embed_dim)

        self.pose_fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # 4. Cross-attention: Pose as query
        self.pose_vision_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pose_vision_ln = nn.LayerNorm(embed_dim)

        # 5. Temporal transformer across frames
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 6. Contact head
        self.contact_head = nn.Linear(embed_dim, output_dim)
    
    def forward(self, image, depth, hand_joint_pose, desired_hand_joint_pose):
        """
            image: (B, Hr, H, W, C)
            depth: (B, Hr, H, W, 1)
            hand_joint_pose: (B, Hr, 16)
            desired_hand_joint_pose: (B, Hr, 16)
        """

        B, H, h, w, c = image.shape

        # Apply the segmentation mask to the image and depth, apply transform and get vision features from AsymFormer 

        # image, depth, segmask = image.to(self.device), depth.to(self.device), segmask.to(self.device)
        # masked_image = (image * segmask).permute(0,1,4,2,3).view(B*H, c, h, w) # masked_image: (B, Hr, H, W, C) -> (B*Hr, C, H, W) -> (B*Hr, C, H, W)
        # masked_depth = (depth * segmask).permute(0,1,4,2,3).view(B*H, 1, h, w) # masked_depth: (B, Hr, H, W, 1) -> (B*Hr, 1, H, W) -> (B*Hr, 1, H, W)
        masked_image = image.permute(0,1,4,2,3).view(B*H, c, h, w) # masked_image: (B, Hr, H, W, C) -> (B*Hr, C, H, W) -> (B*Hr, C, H, W)
        masked_depth = depth.permute(0,1,4,2,3).view(B*H, 1, h, w) # masked_depth: (B, Hr, H, W, 1) -> (B*Hr, 1, H, W)
        
        sample = {'image': masked_image, 'depth': masked_depth} # sample: dict with image: (B*Hr, C, H, W), depth: (B*Hr, 1, H, W)
        sample = transform(sample)
        masked_image, masked_depth = sample['image'].to(self.device), sample['depth'].to(self.device) # image: (B*Hr, C, H, W), depth: (B*Hr, 1, H, W)
        vision_features = self.asymformer(masked_image, masked_depth)

        # _,d1,d2,d3 = vision_features.shape
        # vision_features = vision_features.view(B, H, d1, d2, d3) # vision_features: (B, Hr, D, H', W')

        # # Flatten the (B, Hr) dims for CNN downsampling
        # feats_2d = vision_features.view(B*Hr, c, h, w)

        # 1. Convolutional downsampling -> (B*Hr, new_hw, embed_dim)
        conv_tokens = self.downsampler(vision_features) # e.g. shape (B*Hr, 1200, 256)


        # 2. (optional) Self-attention among visual tokens
        if self.use_vision_attn:
            conv_tokens = self.vision_attn(conv_tokens)

        # 3. Pose embedding -> (B, Hr, embed_dim)
        pose_emb = self.pose_emb(hand_joint_pose)
        desired_pose_emb = self.desired_pose_emb(desired_hand_joint_pose)

        # Flatten for cross-attention: (B, Hr, embed_dim) -> (B*Hr, 1, embed_dim)
        pose_flat = pose_emb.view(B*H, 1, -1)
        desired_pose_flat = desired_pose_emb.view(B*H, 1, -1)

        # We want cross-attention:
        # - Query: pose_flat, shape (B*Hr, 1, embed_dim)
        # - Key, Value: conv_tokens, shape (B*Hr, new_hw, embed_dim)
        pose_attn_out, _ = self.pose_vision_cross_attn(pose_flat, conv_tokens, conv_tokens) # (B*Hr, 1, embed_dim)
        desired_pose_attn_out, _ = self.pose_vision_cross_attn(desired_pose_flat, conv_tokens, conv_tokens) # (B*Hr, 1, embed_dim)

        # Concatenate and fuse: (B*Hr, 1, 2*embed_dim) -> (B*Hr, 1, embed_dim)
        fused = self.pose_fusion_mlp(torch.cat([pose_attn_out, desired_pose_attn_out], dim=-1))

        # Residual + LayerNorm
        fused = self.pose_vision_ln(fused + pose_flat) # shape (B*Hr, 1, embed_dim)

        # Reshape back to (B, Hr, embed_dim)
        fused_frames = fused.view(B, H, -1)

        # 5. CAUSAL temporal self-attention
        #   create a mask: shape (H, H) => each position can't attend future positions
        causal_mask = generate_causal_mask(H, device=fused_frames.device)
        
        # pass mask=causal_mask to TransformerEncoder
        temporal_out = self.temporal_transformer(fused_frames, mask=causal_mask) # shape (B, Hr, embed_dim)
        # temporal_out = self.temporal_transformer(fused_frames) # shape (B, Hr, embed_dim)

        # 6. Contact head
        contact_preds = self.contact_head(temporal_out) # shape (B, Hr, 70)

        return contact_preds