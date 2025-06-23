# --------------------------------------------------------
# Multi-Channel MultiSpecVision Model
# Copyright (c) 2024 MultiSpecVision Team
# Licensed under The MIT License [see LICENSE for details]
# Based on the original MultiSpecVision architecture
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.multispec_transformer import Mlp, WindowAttention, SwinTransformerBlock, window_partition, window_reverse


class DynamicChannelPatchEmbed(nn.Module):
    """
    Dynamic channel Image to Patch Embedding
    Supports 3-20 input channels
    """
    def __init__(self, img_size=224, patch_size=4, in_chans_min=3, in_chans_max=20, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans_min = in_chans_min
        self.in_chans_max = in_chans_max
        self.embed_dim = embed_dim
        
        # Create multiple convolution layers to support different input channel numbers
        self.proj_layers = nn.ModuleDict()
        for i in range(in_chans_min, in_chans_max + 1):
            self.proj_layers[f'chan_{i}'] = nn.Conv2d(i, embed_dim, kernel_size=patch_size, stride=patch_size)
            
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure input image size is correct
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        
        # Ensure channel number is within supported range
        assert self.in_chans_min <= C <= self.in_chans_max, \
            f"Input channel number {C} is out of supported range [{self.in_chans_min}, {self.in_chans_max}]"
        
        # Select projection layer for corresponding channel number
        proj = self.proj_layers[f'chan_{C}']
        x = proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        
        if self.norm is not None:
            x = self.norm(x)
        return x


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism for enhancing multi-channel feature fusion
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim // 4, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B, L, C
        # Global average pooling
        avg_pool = x.mean(dim=1, keepdim=True)  # B, 1, C
        
        # MLP + Sigmoid activation
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.act(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        # Channel attention weights
        channel_weights = self.sigmoid(avg_pool)  # B, 1, C
        
        # Weighted features
        return x * channel_weights


class PatchMerging(nn.Module):
    """ Patch Merging Layer (same as original MultiSpecVision)
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic MultiSpecVision layer (similar to original, but with added channel attention)
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_channel_attn=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_channel_attn = use_channel_attn

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # Channel attention (optional)
        if use_channel_attn:
            self.channel_attn = ChannelAttention(dim)
        
        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        # Apply channel attention
        if self.use_channel_attn:
            x = self.channel_attn(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling module for segmentation decoder - fixed version
    """
    def __init__(self, in_features, out_features, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Normalization layer
        self.norm = norm_layer(in_features)
        
        # Projection layer - always created, even if in_features == out_features
        self.proj = nn.Linear(in_features, out_features)
        
        # Activation function
        self.act = nn.GELU()
        
        print(f"Creating UpsampleBlock: in_features={in_features}, out_features={out_features}")
        
    def forward(self, x, output_size):
        """
        x: B, L, C
        output_size: (H, W)
        """
        B, L, C = x.shape
        H, W = output_size
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply projection to match target feature dimension
        x = self.proj(x)
        
        # Check if sequence length can be reshaped into a square
        side_len = int(L**0.5)
        if side_len * side_len != L:
            # Use adaptive pooling to adjust sequence length
            x = x.permute(0, 2, 1)  # [B, C, L]
            target_len = side_len * side_len
            x = F.adaptive_avg_pool1d(x, target_len)
            x = x.permute(0, 2, 1)  # [B, L', C]
            L = target_len
            
        # Reshape to 2D feature map
        x = x.reshape(B, side_len, side_len, -1)
        
        # Convert to BCHW format for upsampling
        x = x.permute(0, 3, 1, 2)  # [B, C, H', W']
        
        # Upsample to target size
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        
        # Convert back to sequence form
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, H*W, -1)
        
        # Apply activation
        x = self.act(x)
        
        # Handle NaN values
        if torch.isnan(x).any():
            print("[UpsampleBlock] Warning: Output contains NaN values, replacing with 0")
            x = torch.nan_to_num(x, nan=0.0)
        
        return x


class MultiChannelSwinTransformer(nn.Module):
    """
    Multi-channel Swin Transformer model, supports image segmentation
    """
    def __init__(self, img_size=224, patch_size=4, in_chans_min=3, in_chans_max=20, 
                 num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, use_channel_attn=True, task='classification', **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.task = task  # 'classification' or 'segmentation'

        # Split image into non-overlapping patches, supports dynamic channel numbers
        self.patch_embed = DynamicChannelPatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans_min=in_chans_min, in_chans_max=in_chans_max, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Random depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Random depth decay rule

        # Build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               use_channel_attn=use_channel_attn)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        
        # Classification head or segmentation head
        if task == 'classification':
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        else:  # segmentation
            # Decoder layers (for segmentation)
            self.decoder_layers = nn.ModuleList()
            
            # Create dimension adapter dictionary
            self.dim_adapters = nn.ModuleDict()
            
            # Create dimension adapter for each layer
            for i_layer in range(self.num_layers - 1, -1, -1):
                # Calculate current layer and previous layer feature dimension
                if i_layer == self.num_layers - 1:
                    in_dim = self.num_features  # Output dimension of last encoder layer
                else:
                    in_dim = int(embed_dim * 2 ** (i_layer + 1))  # Dimension of previous layer
                
                out_dim = int(embed_dim * 2 ** i_layer)  # Dimension of current layer
                
                # Create dimension adapter
                adapter_name = f"adapter_{in_dim}_{out_dim}"
                self.dim_adapters[adapter_name] = nn.Linear(in_dim, out_dim)
                print(f"Creating dimension adapter: {adapter_name} ({in_dim} -> {out_dim})")
                
                # Create additional adapter for skip connections
                if i_layer < self.num_layers - 1:
                    skip_in_dim = int(embed_dim * 2 ** i_layer)  # Input dimension of skip connection
                    skip_out_dim = out_dim  # Output dimension to match
                    
                    if skip_in_dim != skip_out_dim:
                        skip_adapter_name = f"adapter_{skip_in_dim}_{skip_out_dim}"
                        if skip_adapter_name not in self.dim_adapters:
                            self.dim_adapters[skip_adapter_name] = nn.Linear(skip_in_dim, skip_out_dim)
                            print(f"Creating skip connection adapter: {skip_adapter_name} ({skip_in_dim} -> {skip_out_dim})")
                
                # Create decoder
                self.decoder_layers.append(
                    UpsampleBlock(out_dim, out_dim, norm_layer=norm_layer)
                )
            
            # Final segmentation head
            self.seg_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_classes)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Store intermediate features for skip connections
        features = []
        resolutions = []
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                features.append(x)
                resolutions.append(
                    (self.patches_resolution[0] // (2 ** i), 
                     self.patches_resolution[1] // (2 ** i))
                )
            x = layer(x)

        x = self.norm(x)
        return x, features, resolutions

    def forward(self, x):
        print(f"[DEBUG] Model input shape: {x.shape}")
        x, features, resolutions = self.forward_features(x)
        print(f"[DEBUG] Feature extraction shape: {x.shape}")
        print(f"[DEBUG] Feature list: {[f.shape for f in features]}")
        print(f"[DEBUG] Resolution list: {resolutions}")
        
        if self.task == 'classification':
            # Classification task
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x
        else:
            # Segmentation task
            B, L, C = x.shape
            H, W = self.patches_resolution[0], self.patches_resolution[1]
            print(f"[DEBUG] Segmentation task parameters: B={B}, L={L}, C={C}, H={H}, W={W}")
            
            # Decoder path
            for i, decoder in enumerate(self.decoder_layers):
                print(f"[DEBUG] Decoder {i} starts processing, current feature shape: {x.shape}")
                
                # First apply dimension adapter
                if i == 0:
                    # First decoder, use last layer feature
                    in_dim = self.num_features
                else:
                    # Other decoders, calculate input dimension
                    in_dim = int(self.embed_dim * 2 ** (self.num_layers - i))
                
                out_dim = int(self.embed_dim * 2 ** (self.num_layers - i - 1))
                adapter_name = f"adapter_{in_dim}_{out_dim}"
                
                if adapter_name in self.dim_adapters:
                    print(f"[DEBUG] Applying dimension adapter: {adapter_name}, {x.shape[2]} -> {out_dim}")
                    # Ensure current feature dimension matches input dimension of adapter
                    if x.shape[2] != in_dim:
                        print(f"[WARNING] Feature dimension {x.shape[2]} does not match input dimension {in_dim}, creating temporary adapter")
                        temp_adapter = nn.Linear(x.shape[2], out_dim).to(x.device)
                        x = temp_adapter(x)
                    else:
                        x = self.dim_adapters[adapter_name](x)
                    print(f"[DEBUG] Dimension adapter applied: {x.shape}")
                
                # Skip connection (from second decoder onwards)
                if i > 0 and i <= len(features):
                    # Safely get feature index
                    feat_idx = len(features) - i
                    print(f"[DEBUG] Trying skip connection, feature index: {feat_idx}/{len(features)-1}")
                    
                    if feat_idx >= 0 and feat_idx < len(features):
                        skip_feature = features[feat_idx]
                        print(f"[DEBUG] Skip connection feature shape: {skip_feature.shape}, current feature shape: {x.shape}")
                        
                        # Check and adjust sequence length
                        if skip_feature.shape[1] != x.shape[1]:
                            print(f"[DEBUG] Sequence length mismatch: {skip_feature.shape[1]} vs {x.shape[1]}")
                            # Use adaptive pooling to adjust sequence length
                            skip_feature = skip_feature.permute(0, 2, 1)  # B, C, L
                            skip_feature = F.adaptive_avg_pool1d(skip_feature, x.shape[1])
                            skip_feature = skip_feature.permute(0, 2, 1)  # B, L, C
                            print(f"[DEBUG] Adjusted sequence length: {skip_feature.shape[1]}")
                        
                        # Check and adjust channel dimension
                        if skip_feature.shape[2] != x.shape[2]:
                            print(f"[DEBUG] Channel dimension mismatch: {skip_feature.shape[2]} vs {x.shape[2]}")
                            # Create temporary adapter
                            skip_adapter = nn.Linear(skip_feature.shape[2], x.shape[2]).to(x.device)
                            skip_feature = skip_adapter(skip_feature)
                            print(f"[DEBUG] Adjusted channel dimension: {skip_feature.shape[2]}")
                        
                        # Safely add skip connection
                        x_before = x.clone()
                        x = x + skip_feature
                        
                        # Check for NaN values
                        if torch.isnan(x).any():
                            print("[WARNING] Skip connection resulted in NaN values, restoring original feature")
                            x = x_before
                
                # Determine upsampling target resolution
                if i < len(self.decoder_layers) - 1:
                    # Use resolution of corresponding encoder layer
                    res_idx = len(resolutions) - i - 1
                    if res_idx >= 0 and res_idx < len(resolutions):
                        next_res = resolutions[res_idx]
                    else:
                        # If index out of range, use 2x resolution of previous layer
                        h = H // (2 ** (self.num_layers - i - 1))
                        w = W // (2 ** (self.num_layers - i - 1))
                        next_res = (h, w)
                else:
                    # Last layer uses original resolution
                    next_res = (H, W)
                
                print(f"[DEBUG] Upsampling target resolution: {next_res}")
                
                try:
                    x = decoder(x, next_res)
                    print(f"[DEBUG] Decoder {i} output shape: {x.shape}")
                except RuntimeError as e:
                    print(f"[ERROR] Decoder {i} error: {str(e)}")
                    # Emergency handling
                    # 1. Adjust feature dimension
                    if "size mismatch" in str(e) and hasattr(decoder, 'proj'):
                        print(f"[DEBUG] Trying direct feature dimension adjustment")
                        x = decoder.norm(x)
                        x = decoder.proj(x)
                    
                    # 2. Force adjust sequence length
                    x = x.permute(0, 2, 1)  # B, C, L
                    target_len = next_res[0] * next_res[1]
                    x = F.adaptive_avg_pool1d(x, target_len)
                    x = x.permute(0, 2, 1)  # B, L, C
                    print(f"[DEBUG] Emergency adjusted feature shape: {x.shape}")
            
            # Segmentation head
            print(f"[DEBUG] Applying segmentation head before shape: {x.shape}")
            x = self.seg_head(x)  # B, H*W, num_classes
            print(f"[DEBUG] Segmentation head output shape: {x.shape}")
            
            # Reshape to image size
            try:
                x = x.permute(0, 2, 1).reshape(B, self.num_classes, H, W)
                print(f"[DEBUG] Final output shape: {x.shape}")
            except RuntimeError as e:
                print(f"[ERROR] Reshape error: {str(e)}")
                # Use adaptive pooling to adjust size
                x = x.permute(0, 2, 1)  # B, num_classes, L
                x = F.adaptive_avg_pool1d(x, H*W)
                x = x.reshape(B, self.num_classes, H, W)
                print(f"[DEBUG] Adjusted final output shape: {x.shape}")
            
            # Check if output contains NaN values
            if torch.isnan(x).any():
                print("[ERROR] Output contains NaN values, replacing with 0")
                x = torch.nan_to_num(x, nan=0.0)
            
            return x

    def load_from(self, pretrained_model_path, in_chans=3):
        """
        Load weights from a pre-trained Swin Transformer model
        """
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
                
            # Load main network weights
            state_dict = self.state_dict()
            
            # Skip classification head and segmentation head
            skip_keys = ['head.weight', 'head.bias', 'seg_head']
            for k in list(checkpoint_model.keys()):
                if any(sk in k for sk in skip_keys):
                    print(f"Skipping {k} (classification head/segmentation head)")
                    checkpoint_model.pop(k, None)
                    
            # Process patch_embed layer, only load weights for matching channel numbers
            if f'patch_embed.proj.weight' in checkpoint_model:
                pretrained_proj = checkpoint_model['patch_embed.proj.weight']
                if pretrained_proj.shape[1] == in_chans:
                    key = f'patch_embed.proj_layers.chan_{in_chans}.weight'
                    if key in state_dict:
                        state_dict[key] = pretrained_proj
                        print(f"Loading patch_embed weights: {pretrained_proj.shape}")
                        if f'patch_embed.proj.bias' in checkpoint_model:
                            bias_key = f'patch_embed.proj_layers.chan_{in_chans}.bias'
                            if bias_key in state_dict:
                                state_dict[bias_key] = checkpoint_model['patch_embed.proj.bias']
                    else:
                        print(f"Warning: Target key {key} not found")
                else:
                    print(f"Warning: patch_embed.proj channel number mismatch - Pretrained: {pretrained_proj.shape[1]}, Target: {in_chans}")
                        
            # Load other layers
            loaded_count = 0
            skipped_count = 0
            for k in checkpoint_model.keys():
                if k.startswith('patch_embed.proj'):
                    continue  # Already processed
                    
                if k in state_dict:
                    if checkpoint_model[k].shape == state_dict[k].shape:
                        state_dict[k] = checkpoint_model[k]
                        loaded_count += 1
                    else:
                        print(f"Warning: Skipping {k} because shape mismatch - Pretrained: {checkpoint_model[k].shape}, Target: {state_dict[k].shape}")
                        skipped_count += 1
            
            # Load compatible weights
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"Loading pre-trained weights completed: Successfully loaded {loaded_count} layers, skipped {skipped_count} layers")
            print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            
            return self
        except Exception as e:
            print(f"Loading pre-trained weights failed: {str(e)}")
            print("Using random initialization weights")
            return self 