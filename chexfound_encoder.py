import os
import torch
import torch.nn as nn
import argparse
from typing import Optional, Tuple, Union, List
from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform
from PIL import Image


class CheXFoundEncoder(nn.Module):
    """
    CheXFound encoder wrapper for feature extraction from chest X-ray images.
    
    CheXFound is a foundation model trained with DINOv2-like self-supervised learning
    specifically for chest X-ray analysis.
    """
    
    def __init__(
        self,
        config_file: str = "/mnt/data/checkpoints/chexfound/config.yaml",
        pretrained_weights: str = "/mnt/data/checkpoints/chexfound/teacher_checkpoint.pth",
        image_size: int = 512,
        patch_size: int = 16,
        n_register_tokens: int = 4,
        n_last_blocks: int = 4,
        return_class_token: bool = True,
        num_classes: int = 40,
        num_heads: int = 8,
        device: str = "cuda"
    ):
        """
        Initialize CheXFound encoder.
        
        Args:
            config_file: Path to architecture configuration file
            pretrained_weights: Path to pretrained model weights
            image_size: Input image size (default: 512)
            patch_size: Patch size for vision transformer (default: 16)
            n_register_tokens: Number of register tokens (default: 4)
            n_last_blocks: Number of last blocks to use for features (default: 4)
            return_class_token: Whether to return class token (default: True)
            num_classes: Number of output classes (default: 40)
            num_heads: Number of attention heads (default: 8)
            device: Device to run the model on
        """
        super().__init__()
        
        self.config_file = config_file
        self.pretrained_weights = pretrained_weights
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_register_tokens = n_register_tokens
        self.n_last_blocks = n_last_blocks
        self.return_class_token = return_class_token
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.device = device
        
        # Initialize the model
        self._setup_model()
        self._load_pretrained_weights()
        
        # Setup data transform
        self.eval_transform = make_classification_eval_transform(
            resize_size=self.image_size, 
            crop_size=self.image_size
        )
        
    def _setup_model(self):
        """Setup the foundation model architecture."""
        parser = argparse.ArgumentParser()
        parser.set_defaults(
            config_file=self.config_file,
            pretrained_weights=None,
            output_dir="./output",
            opts=[],
            image_size=self.image_size,
            patch_size=self.patch_size,
            n_register_tokens=self.n_register_tokens,
            n_last_blocks=self.n_last_blocks,
            return_class_token=self.return_class_token,
            num_classes=self.num_classes,
            num_heads=self.num_heads,
        )
        self.args, _ = parser.parse_known_args()
        
        # Setup foundation model
        self.model, self.autocast_dtype = setup_and_build_model(self.args)
        self.model = self.model.to(self.device)
        
    def _load_pretrained_weights(self):
        """Load pretrained weights with proper key mapping."""
        if not os.path.exists(self.pretrained_weights):
            raise FileNotFoundError(f"Pretrained weights not found at {self.pretrained_weights}")
            
        state_dict = torch.load(self.pretrained_weights, map_location=self.device)['teacher']
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                ls = k.split('.')
                if 'blocks' in k:
                    new_k = '.'.join([ls[1], *ls[3:]])
                else:
                    new_k = '.'.join(ls[1:])
            else:
                new_k = k
            new_state_dict.update({new_k: v})
        
        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {self.pretrained_weights}")
        
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        img = img.convert(mode="RGB")
        img = self.eval_transform(img)
        return img.to(self.device)
        
    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract features from input images.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (patch_features, cls_features):
            - patch_features: List of patch token tensors from the last n blocks
            - cls_features: List of class token tensors from the last n blocks
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model.get_intermediate_layers(
                x,
                n=self.n_last_blocks,
                return_class_token=self.return_class_token,
            )
        
        # Separate patch tokens and class tokens
        patch_features = []
        cls_features = []
        
        for feature_tensor in features:
            if self.return_class_token:
                cls_token = feature_tensor[1]  # [batch_size, feature_dim]
                patch_tokens = feature_tensor[0] # [batch_size, num_patches, feature_dim]
                
                cls_features.append(cls_token)
                patch_features.append(patch_tokens)
            else:
                # Only patch tokens
                patch_features.append(feature_tensor)
                cls_features.append(None)
        
        return patch_features, cls_features
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the encoder."""
        return self.encode(x)
        
    def encode_image(self, image: Union[str, Image.Image]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a single image from file path or PIL Image.
        
        Args:
            image: Path to the image file or PIL Image object
            
        Returns:
            Tuple of (patch_features, cls_features)
        """
        img = self.preprocess_image(image)
        img = img.unsqueeze(0)  # Add batch dimension
        return self.encode(img)
        
    def encode_batch(self, image_paths: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Tuple of (patch_features, cls_features) for the batch
        """
        batch_imgs = []
        for path in image_paths:
            img = self.preprocess_image(path)
            batch_imgs.append(img)
        
        batch_tensor = torch.stack(batch_imgs, dim=0)
        return self.encode(batch_tensor)
    
    def get_patch_tokens_only(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get only patch tokens (spatial features) without class tokens.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            List of patch token tensors from the last n blocks
        """
        patch_features, _ = self.encode(x)
        return patch_features
    
    def get_cls_tokens_only(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get only class tokens (global features) without patch tokens.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            List of class token tensors from the last n blocks
        """
        _, cls_features = self.encode(x)
        return cls_features
    
    def get_spatial_feature_map(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get spatial feature map from patch tokens.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            layer_idx: Which layer to use (-1 for last layer)
            
        Returns:
            Spatial feature map of shape (B, H_patches, W_patches, feature_dim)
        """
        patch_features, _ = self.encode(x)
        selected_features = patch_features[layer_idx]  # [B, num_patches, feature_dim]
        
        # Reshape to spatial grid
        batch_size, num_patches, feature_dim = selected_features.shape
        h_patches = w_patches = int(num_patches ** 0.5)  # Assuming square patches
        
        spatial_map = selected_features.view(batch_size, h_patches, w_patches, feature_dim)
        return spatial_map
        
    def get_feature_dim(self) -> int:
        """Get the feature dimension of the encoder."""
        # Create a dummy input to get feature dimensions
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            patch_features, cls_features = self.encode(dummy_input)
        return patch_features[0].shape[-1]  # Return the feature dimension


# Example usage and testing
if __name__ == "__main__":
    # Initialize the encoder
    encoder = CheXFoundEncoder(
        config_file="/mnt/data/checkpoints/chexfound/config.yaml",
        pretrained_weights="/mnt/data/checkpoints/chexfound/teacher_checkpoint.pth"
    )
    
    print(f"Feature dimension: {encoder.get_feature_dim()}")
    
    # Example: encode a single image
    # patch_features, cls_features = encoder.encode_image("/path/to/your/chest_xray.jpg")
    # print(f"Number of feature layers: {len(patch_features)}")
    # print(f"Patch features shape: {patch_features[0].shape}")  # [1, num_patches, feature_dim]
    # print(f"Class token shape: {cls_features[0].shape}")      # [1, feature_dim]
    
    # Example: get only patch tokens for spatial analysis
    # dummy_input = torch.randn(2, 3, 512, 512).cuda()
    # patch_only = encoder.get_patch_tokens_only(dummy_input)
    # print(f"Patch-only features shape: {patch_only[0].shape}")  # [2, 1024, feature_dim]
    
    # Example: get spatial feature map
    # spatial_map = encoder.get_spatial_feature_map(dummy_input)
    # print(f"Spatial map shape: {spatial_map.shape}")  # [2, 32, 32, feature_dim]