import numpy as np
import os
import rawpy
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DNGProcessor')

class DNGProcessor:
    """
    DNG file processing module for reading and processing DNG format multi-channel images
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        norm_mean: List[float] = [0.485, 0.456, 0.406],
        norm_std: List[float] = [0.229, 0.224, 0.225],
        bit_depth_normalization: bool = True,
        debug: bool = True
    ):
        """
        Initialize DNG processor
        
        Args:
            target_size: Output image size (height, width)
            normalize: Whether to perform normalization
            norm_mean: Normalization mean (for RGB channels, other channels use statistical calculation)
            norm_std: Normalization standard deviation (for RGB channels, other channels use statistical calculation)
            bit_depth_normalization: Whether to normalize based on bit depth
            debug: Whether to print debug information
        """
        self.target_size = target_size
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.bit_depth_normalization = bit_depth_normalization
        self.debug = debug
        
        if self.debug:
            logger.info(f"DNGProcessor initialized: target_size={target_size}, normalize={normalize}")
        
    def read_dng(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Read DNG file, extract raw sensor data and metadata
        
        Args:
            file_path: DNG file path
            
        Returns:
            Dictionary containing different channel data
        """
        if not os.path.exists(file_path):
            error_msg = f"File does not exist: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            logger.info(f"Starting to read DNG file: {file_path}")
            with rawpy.imread(file_path) as raw:
                # Get raw sensor data
                try:
                    raw_data = raw.raw_image.copy()
                    logger.info(f"Raw data shape: {raw_data.shape}, type: {raw_data.dtype}")
                except Exception as e:
                    logger.warning(f"Failed to read raw image data: {str(e)}, trying raw_image_visible")
                    raw_data = raw.raw_image_visible.copy()
                    logger.info(f"Visible raw data shape: {raw_data.shape}, type: {raw_data.dtype}")
                
                # Get metadata
                try:
                    metadata = {
                        'black_level': raw.black_level_per_channel,
                        'white_level': raw.white_level,
                        'color_matrix': raw.color_matrix if hasattr(raw, 'color_matrix') else None,
                        'camera_white_balance': raw.camera_white_balance if hasattr(raw, 'camera_white_balance') else None,
                        'daylight_white_balance': raw.daylight_whitebalance if hasattr(raw, 'daylight_whitebalance') else None,
                        'num_colors': raw.num_colors,
                        'raw_pattern': raw.raw_pattern if hasattr(raw, 'raw_pattern') else None,
                        'raw_colors': raw.raw_colors if hasattr(raw, 'raw_colors') else None,
                    }
                    logger.info(f"Successfully extracted metadata: num_colors={metadata['num_colors']}")
                except Exception as e:
                    logger.warning(f"Failed to extract metadata: {str(e)}, using default values")
                    metadata = {
                        'num_colors': 3,
                        'raw_pattern': np.array([[0, 1], [1, 2]]) if hasattr(np, 'array') else None
                    }
                
                # Process different color channels
                logger.info("Starting RGB image post-processing...")
                try:
                    rgb_image = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=True,
                        output_bps=16
                    )
                    logger.info(f"RGB image post-processing completed: shape={rgb_image.shape}, type={rgb_image.dtype}")
                except Exception as e:
                    logger.error(f"RGB post-processing failed: {str(e)}")
                    # Create a backup RGB image
                    logger.info("Using backup method to create RGB image")
                    if len(raw_data.shape) == 2:
                        # If raw data is single channel, duplicate three times to create RGB
                        rgb_image = np.stack([raw_data] * 3, axis=2)
                    else:
                        # If already multi-channel, ensure there are 3 channels
                        if raw_data.shape[2] >= 3:
                            rgb_image = raw_data[:, :, :3]
                        else:
                            # Insufficient channels, duplicate existing channels
                            channels = [raw_data[:, :, i] for i in range(raw_data.shape[2])]
                            while len(channels) < 3:
                                channels.append(channels[-1])
                            rgb_image = np.stack(channels, axis=2)
                    
                    # Normalize to 16-bit
                    if rgb_image.dtype != np.uint16:
                        max_val = rgb_image.max()
                        if max_val > 0:
                            rgb_image = (rgb_image / max_val * 65535).astype(np.uint16)
                        else:
                            rgb_image = rgb_image.astype(np.uint16)
                    
                    logger.info(f"Backup RGB image created: shape={rgb_image.shape}, type={rgb_image.dtype}")
                
                # Extract other possible channel data
                channels = {
                    'raw': raw_data,
                    'rgb': rgb_image,
                    'metadata': metadata
                }
                
                logger.info("DNG file reading completed")
                return channels
                
        except Exception as e:
            error_msg = f"Error processing DNG file: {str(e)}"
            logger.error(error_msg)
            # Return minimal valid data structure to avoid complete failure
            dummy_data = np.zeros((224, 224), dtype=np.uint16)
            dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint16)
            return {
                'raw': dummy_data,
                'rgb': dummy_rgb,
                'metadata': {'num_colors': 3, 'error': str(e)}
            }
    
    def extract_channels(
        self, 
        dng_data: Dict[str, np.ndarray], 
        channel_selection: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract specified channels from DNG data
        
        Args:
            dng_data: Data dictionary returned from read_dng
            channel_selection: List of channels to extract, if None extract all available channels
            
        Returns:
            Multi-channel image array after channel merging
        """
        available_channels = []
        metadata = dng_data['metadata']
        
        # Extract RGB channels
        if 'rgb' in dng_data:
            rgb = dng_data['rgb']
            r_channel = rgb[:, :, 0]
            g_channel = rgb[:, :, 1]
            b_channel = rgb[:, :, 2]
            
            available_channels.extend([
                ('r', r_channel),
                ('g', g_channel),
                ('b', b_channel)
            ])
        
        # Extract raw sensor data
        if 'raw' in dng_data:
            raw_data = dng_data['raw']
            available_channels.append(('raw', raw_data))
            
            # If there's a color filter array (CFA), separate different color channels
            if metadata['num_colors'] > 1:
                pattern = metadata['raw_pattern']
                height, width = raw_data.shape
                
                # Separate channels based on Bayer pattern
                for color_idx in range(metadata['num_colors']):
                    color_mask = np.zeros_like(raw_data, dtype=bool)
                    
                    # Find pixel positions for corresponding color
                    if pattern is not None and hasattr(pattern, 'shape'):
                        pattern_h, pattern_w = pattern.shape
                        for y in range(0, height, pattern_h):
                            for x in range(0, width, pattern_w):
                                end_y = min(y + pattern_h, height)
                                end_x = min(x + pattern_w, width)
                                local_pattern = pattern[:end_y-y, :end_x-x]
                                color_mask[y:end_y, x:end_x] = (local_pattern == color_idx)
                    
                    if color_mask.any():
                        color_channel = np.zeros_like(raw_data)
                        color_channel[color_mask] = raw_data[color_mask]
                        available_channels.append((f'cfa_{color_idx}', color_channel))
        
        # Apply channel selection
        if channel_selection is not None:
            selected_channels = []
            for channel_name in channel_selection:
                for name, data in available_channels:
                    if name == channel_name:
                        selected_channels.append(data)
                        break
            if not selected_channels:
                logger.warning(f"No channels found matching selection: {channel_selection}")
                selected_channels = [available_channels[0][1]] if available_channels else [np.zeros((224, 224))]
        else:
            selected_channels = [data for _, data in available_channels]
        
        if not selected_channels:
            logger.warning("No channels available, returning zeros")
            return np.zeros((224, 224, 1))
        
        # Ensure all channels have same dimensions
        target_shape = selected_channels[0].shape[:2]
        normalized_channels = []
        
        for channel in selected_channels:
            if len(channel.shape) == 3:
                # If channel is RGB, convert to grayscale
                channel = np.mean(channel, axis=2)
            
            if channel.shape[:2] != target_shape:
                # Resize channel to match target shape
                channel_pil = Image.fromarray(channel.astype(np.uint16))
                channel_pil = channel_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
                channel = np.array(channel_pil)
            
            normalized_channels.append(channel)
        
        # Stack channels
        if len(normalized_channels) == 1:
            multi_channel = normalized_channels[0][:, :, np.newaxis]
        else:
            multi_channel = np.stack(normalized_channels, axis=2)
        
        logger.info(f"Extracted {multi_channel.shape[2]} channels, final shape: {multi_channel.shape}")
        return multi_channel
    
    def preprocess(
        self, 
        image: np.ndarray, 
        bit_depth: int = 16
    ) -> torch.Tensor:
        """
        Preprocess multi-channel image for model input
        
        Args:
            image: Multi-channel image array
            bit_depth: Bit depth of input image
            
        Returns:
            Preprocessed tensor
        """
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        height, width, channels = image.shape
        logger.info(f"Preprocessing image: shape={image.shape}, channels={channels}, bit_depth={bit_depth}")
        
        # Resize to target size
        if (height, width) != self.target_size:
            resized_channels = []
            for c in range(channels):
                channel = image[:, :, c]
                if channel.dtype == np.uint16:
                    channel_pil = Image.fromarray(channel, mode='I;16')
                else:
                    channel_pil = Image.fromarray(channel.astype(np.uint8))
                
                channel_pil = channel_pil.resize(
                    (self.target_size[1], self.target_size[0]), 
                    Image.LANCZOS
                )
                resized_channels.append(np.array(channel_pil))
            
            image = np.stack(resized_channels, axis=2)
        
        # Normalize based on bit depth
        if self.bit_depth_normalization:
            max_val = (2 ** bit_depth) - 1
            image = image.astype(np.float32) / max_val
        else:
            # Use statistical normalization
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / image.max()
        
        # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # Apply normalization if requested
        if self.normalize:
            if channels == 3:
                # Use ImageNet normalization for RGB
                normalize_transform = transforms.Normalize(
                    mean=self.norm_mean,
                    std=self.norm_std
                )
                tensor = normalize_transform(tensor)
            else:
                # Use per-channel normalization for multi-spectral data
                for c in range(channels):
                    channel_data = tensor[c]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    if std > 0:
                        tensor[c] = (channel_data - mean) / std
        
        logger.info(f"Preprocessing completed: output shape={tensor.shape}")
        return tensor
    
    def process_dng_file(
        self, 
        file_path: str, 
        channel_selection: Optional[List[str]] = None,
        bit_depth: int = 16
    ) -> torch.Tensor:
        """
        Process complete DNG file from reading to preprocessing
        
        Args:
            file_path: DNG file path
            channel_selection: Channels to extract
            bit_depth: Bit depth assumption
            
        Returns:
            Preprocessed tensor ready for model input
        """
        logger.info(f"Starting to process DNG file: {file_path}")
        
        # Read DNG file
        dng_data = self.read_dng(file_path)
        
        # Extract channels
        multi_channel_image = self.extract_channels(dng_data, channel_selection)
        
        # Preprocess
        tensor = self.preprocess(multi_channel_image, bit_depth)
        
        logger.info(f"DNG file processing completed: {file_path}")
        return tensor
    
    def get_available_channels(self, file_path: str) -> List[str]:
        """
        Get list of available channels in DNG file
        
        Args:
            file_path: DNG file path
            
        Returns:
            List of available channel names
        """
        try:
            dng_data = self.read_dng(file_path)
            metadata = dng_data['metadata']
            
            channels = ['r', 'g', 'b', 'raw']
            
            # Add CFA channels if available
            if metadata['num_colors'] > 1:
                for i in range(metadata['num_colors']):
                    channels.append(f'cfa_{i}')
            
            return channels
        except Exception as e:
            logger.error(f"Error getting available channels: {str(e)}")
            return ['r', 'g', 'b'] 