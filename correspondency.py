"""
CheXFound Image Correspondence Matcher

This module provides functionality for finding correspondences between chest X-ray images
using the CheXFound encoder.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.figure import Figure
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from chexfound_encoder import CheXFoundEncoder

# Type aliases for better readability
Position = Tuple[int, int]
GridSize = Tuple[int, int]
ImageArray = np.ndarray
Features = np.ndarray


@dataclass
class TransformInfo:
    """Stores image transformation parameters for coordinate conversion."""
    scale_factor: float
    crop_top: int
    crop_left: int
    original_height: int
    original_width: int


@dataclass
class MatchResult:
    """Represents a single match result."""
    position: Position
    token_idx: int
    distance: float


@dataclass
class CorrespondenceResult:
    """Represents correspondence results for a query position."""
    query_position: Position
    query_token_idx: int
    matches: List[MatchResult]


class CheXFoundMatcher:
    """
    Enhanced CheXFound matcher with improved architecture and type safety.
    
    This class provides functionality for finding correspondences between chest X-ray images
    using the CheXFound encoder with streamlined methods and better error handling.
    """
    
    def __init__(
        self,
        config_file: str = "/mnt/data/checkpoints/chexfound/config.yaml",
        pretrained_weights: str = "/mnt/data/checkpoints/chexfound/teacher_checkpoint.pth",
        device: str = "cuda:0"
    ) -> None:
        """
        Initialize the CheXFound matcher.
        
        Args:
            config_file: Path to CheXFound config file
            pretrained_weights: Path to pretrained weights
            device: Device to run inference on
        """
        self.device = device
        self.encoder = CheXFoundEncoder(
            config_file=config_file,
            pretrained_weights=pretrained_weights,
            device=device
        )
        
        # CheXFound configuration constants
        self.IMAGE_SIZE = 512
        self.PATCH_SIZE = 16
        self.NUM_TOKENS = (self.IMAGE_SIZE // self.PATCH_SIZE) ** 2  # 1024 tokens
        self.GRID_SIZE = int(np.sqrt(self.NUM_TOKENS))  # 32x32 grid

    def _prepare_image(self, image: ImageArray) -> Tuple[Image.Image, GridSize, TransformInfo]:
        """
        Prepare image for CheXFound processing with proper transformations.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (PIL Image, grid size, transformation info)
        """
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        original_height, original_width = image.shape[:2]
        
        # Calculate resize and crop parameters
        scale_factor = self.IMAGE_SIZE / min(original_height, original_width)
        resized_height = int(original_height * scale_factor)
        resized_width = int(original_width * scale_factor)
        
        crop_top = (resized_height - self.IMAGE_SIZE) // 2
        crop_left = (resized_width - self.IMAGE_SIZE) // 2
        
        transform_info = TransformInfo(
            scale_factor=scale_factor,
            crop_top=crop_top,
            crop_left=crop_left,
            original_height=original_height,
            original_width=original_width
        )
        
        return pil_image, (self.GRID_SIZE, self.GRID_SIZE), transform_info

    def _extract_features(self, image: Union[ImageArray, Image.Image]) -> Features:
        """
        Extract features using CheXFound encoder.
        
        Args:
            image: Input image
            
        Returns:
            Feature array of shape [num_patches, feature_dim]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        patch_features, _ = self.encoder.encode_image(image)
        return patch_features[-1].squeeze(0).cpu().float().numpy()

    def _position_to_token_idx(self, position: Position, grid_size: GridSize, transform_info: TransformInfo) -> int:
        """
        Convert image position to token index.
        
        Args:
            position: (row, col) position in original image
            grid_size: Token grid dimensions
            transform_info: Transformation parameters
            
        Returns:
            Token index
        """
        row, col = position
        
        # Apply transformations: resize then crop
        resized_row = row * transform_info.scale_factor
        resized_col = col * transform_info.scale_factor
        
        processed_row = resized_row - transform_info.crop_top
        processed_col = resized_col - transform_info.crop_left
        
        # Convert to token indices with bounds checking
        token_row = max(0, min(int(processed_row // self.PATCH_SIZE), grid_size[0] - 1))
        token_col = max(0, min(int(processed_col // self.PATCH_SIZE), grid_size[1] - 1))
        
        return token_row * grid_size[1] + token_col

    def _token_idx_to_position(self, token_idx: int, grid_size: GridSize, transform_info: TransformInfo) -> Position:
        """
        Convert token index to original image position.
        
        Args:
            token_idx: Token index
            grid_size: Token grid dimensions
            transform_info: Transformation parameters
            
        Returns:
            (row, col) position in original image
        """
        # Convert token index to processed image coordinates
        processed_row = (token_idx // grid_size[1]) * self.PATCH_SIZE + self.PATCH_SIZE / 2
        processed_col = (token_idx % grid_size[1]) * self.PATCH_SIZE + self.PATCH_SIZE / 2
        
        # Reverse transformations: add crop offset then divide by scale
        resized_row = processed_row + transform_info.crop_top
        resized_col = processed_col + transform_info.crop_left
        
        original_row = int(resized_row / transform_info.scale_factor)
        original_col = int(resized_col / transform_info.scale_factor)
        
        return original_row, original_col

    def _build_knn_matcher(self, features: Features, k_neighbors: int) -> NearestNeighbors:
        """Build and fit KNN matcher for features."""
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
        nbrs.fit(features)
        return nbrs

    def find_correspondences(
        self,
        query_image: ImageArray,
        target_image: ImageArray,
        query_positions: List[Position],
        k_neighbors: int = 1
    ) -> Tuple[List[CorrespondenceResult], Tuple[GridSize, TransformInfo, GridSize, TransformInfo]]:
        """
        Find correspondences for specific query positions between two images.
        
        Query positions are always in the first image (query_image), and matches 
        are found in the second image (target_image).
        
        Args:
            query_image: Image containing the query positions
            target_image: Image to search for matches
            query_positions: List of (row, col) positions in query_image
            k_neighbors: Number of nearest neighbors to find
            
        Returns:
            Tuple of (correspondence results, grid info)
        """
        print(f"Finding correspondences for {len(query_positions)} query positions...")
        
        # Extract features from both images
        _, query_grid_size, query_transform_info = self._prepare_image(query_image)
        query_features = self._extract_features(query_image)
        
        _, target_grid_size, target_transform_info = self._prepare_image(target_image)
        target_features = self._extract_features(target_image)
        
        # Build KNN matcher with target features
        knn_matcher = self._build_knn_matcher(target_features, k_neighbors)
        
        correspondences = []
        for query_position in query_positions:
            # Convert position to token index
            token_idx = self._position_to_token_idx(query_position, query_grid_size, query_transform_info)
            
            if token_idx >= len(query_features):
                print(f"Warning: Position {query_position} is out of bounds, skipping...")
                continue
            
            # Find nearest neighbors
            query_feature = query_features[token_idx].reshape(1, -1)
            distances, indices = knn_matcher.kneighbors(query_feature)
            
            # Convert results to match objects
            matches = []
            for i in range(k_neighbors):
                match_token_idx = indices[0, i]
                match_distance = distances[0, i]
                match_position = self._token_idx_to_position(
                    match_token_idx, target_grid_size, target_transform_info
                )
                
                matches.append(MatchResult(
                    position=match_position,
                    token_idx=match_token_idx,
                    distance=match_distance
                ))
            
            correspondences.append(CorrespondenceResult(
                query_position=query_position,
                query_token_idx=token_idx,
                matches=matches
            ))
        
        return correspondences, (query_grid_size, query_transform_info, target_grid_size, target_transform_info)

    def visualize_correspondences(
        self,
        query_image: ImageArray,
        target_image: ImageArray,
        correspondences: List[CorrespondenceResult],
        title1: str = "Query Image",
        title2: str = "Target Image",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Visualize correspondences between two images.
        
        Args:
            query_image: Image containing query positions
            target_image: Image containing matched positions
            correspondences: Correspondence results
            title1, title2: Subplot titles
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(query_image)
        ax1.set_title(title1)
        ax2.imshow(target_image)
        ax2.set_title(title2)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(correspondences)))
        
        for i, (corr, color) in enumerate(zip(correspondences, colors)):
            query_row, query_col = corr.query_position
            
            # Plot query points on the first image
            ax1.plot(query_col, query_row, 'o', color=color, markersize=12,
                    markeredgecolor='white', markeredgewidth=3, label=f'Query {i}')
            ax1.text(query_col, query_row - 20, f'Q{i}', fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Plot matches on the second image
            for j, match in enumerate(corr.matches):
                match_row, match_col = match.position
                
                # Different markers for different match ranks
                marker = 's' if j == 0 else ('^' if j == 1 else 'v')
                markersize = 12 - j * 2
                alpha = 1.0 - j * 0.2
                
                ax2.plot(match_col, match_row, marker, color=color, markersize=markersize,
                        markeredgecolor='white', markeredgewidth=2, alpha=alpha)
                
                # Add distance labels
                label_text = f'{i}-{j}\nd={match.distance:.3f}'
                ax2.text(match_col, match_row - 15, label_text, fontsize=8, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
                
                # Draw connection lines
                con = ConnectionPatch((query_col, query_row), (match_col, match_row),
                                    "data", "data", axesA=ax1, axesB=ax2,
                                    color=color, alpha=alpha*0.7, linewidth=3-j)
                
                ax2.add_patch(con)
        
        # Add legend
        self._add_visualization_legend(ax2)
        
        # Set axis limits
        ax1.set_xlim(0, query_image.shape[1])
        ax1.set_ylim(query_image.shape[0], 0)
        ax2.set_xlim(0, target_image.shape[1])
        ax2.set_ylim(target_image.shape[0], 0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    def _add_visualization_legend(self, ax) -> None:
        """Add legend to visualization plot."""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10,
                   markeredgecolor='white', markeredgewidth=2, label='Query points'),
            Line2D([0], [0], marker='s', color='gray', linestyle='None', markersize=10,
                   markeredgecolor='white', markeredgewidth=2, label='Best matches (rank 1)'),
            Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=8,
                   markeredgecolor='white', markeredgewidth=2, label='2nd best matches'),
            Line2D([0], [0], marker='v', color='gray', linestyle='None', markersize=6,
                   markeredgecolor='white', markeredgewidth=2, label='Other matches')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    def analyze_global_correspondences(
        self,
        image1: ImageArray,
        image2: ImageArray,
        num_correspondences: int = 20,
        save_prefix: str = "correspondence"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze global correspondences between two images.
        
        Args:
            image1, image2: Input images
            num_correspondences: Number of top correspondences to visualize
            save_prefix: Prefix for saved files
            
        Returns:
            Tuple of (distances, indices) from KNN analysis
        """
        print("Analyzing global correspondences...")
        
        # Extract features
        features1 = self._extract_features(image1)
        features2 = self._extract_features(image2)
        
        print(f"Feature shapes: {features1.shape}, {features2.shape}")
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features1)
        distances, indices = nbrs.kneighbors(features2)
        
        # Create visualizations
        self._plot_distance_analysis(distances, f'{save_prefix}_distance_analysis.png')
        self._plot_global_correspondences(
            image1, image2, distances, indices, num_correspondences,
            f'{save_prefix}_visualization.png'
        )
        
        # Print statistics
        self._print_correspondence_statistics(distances)
        
        return distances, indices

    def _plot_distance_analysis(self, distances: np.ndarray, save_path: str) -> None:
        """Plot distance distribution analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        
        ax1.hist(distances[:, 0], bins=50, alpha=0.7)
        ax1.set_title("Distance to Nearest Neighbor")
        ax1.set_xlabel("Distance")
        ax1.set_ylabel("Frequency")
        
        ax2.plot(sorted(distances[:, 0]))
        ax2.set_title("Sorted Distances to Nearest Neighbor")
        ax2.set_xlabel("Token Index")
        ax2.set_ylabel("Distance")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_global_correspondences(
        self,
        image1: ImageArray,
        image2: ImageArray,
        distances: np.ndarray,
        indices: np.ndarray,
        num_correspondences: int,
        save_path: str
    ) -> None:
        """Plot global correspondence visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        ax1.imshow(image1)
        ax1.set_title("Image 1 (Source)")
        ax2.imshow(image2)
        ax2.set_title("Image 2 (Target)")
        
        # Get image preparation info for coordinate conversion
        _, grid_size1, transform_info1 = self._prepare_image(image1)
        _, grid_size2, transform_info2 = self._prepare_image(image2)
        
        # Find and plot top correspondences
        top_matches = np.argsort(distances[:, 0])[:num_correspondences]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_correspondences))
        
        for i, (match_idx, color) in enumerate(zip(top_matches, colors)):
            source_idx = indices[match_idx, 0]
            target_idx = match_idx
            
            # Convert to image coordinates
            source_pos = self._token_idx_to_position(source_idx, grid_size1, transform_info1)
            target_pos = self._token_idx_to_position(target_idx, grid_size2, transform_info2)
            
            # Plot points and connections
            ax1.plot(source_pos[1], source_pos[0], 'o', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=2)
            ax2.plot(target_pos[1], target_pos[0], 'o', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=2)
            
            # Add labels
            ax1.text(source_pos[1], source_pos[0], str(i), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            ax2.text(target_pos[1], target_pos[0], str(i), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Connection lines
            con = ConnectionPatch((source_pos[1], source_pos[0]), (target_pos[1], target_pos[0]),
                                "data", "data", axesA=ax1, axesB=ax2,
                                color=color, alpha=0.6, linewidth=2)
            ax2.add_patch(con)
        
        # Set proper axis limits to prevent negative coordinates
        ax1.set_xlim(0, image1.shape[1])
        ax1.set_ylim(image1.shape[0], 0)
        ax2.set_xlim(0, image2.shape[1])
        ax2.set_ylim(image2.shape[0], 0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _print_correspondence_statistics(self, distances: np.ndarray) -> None:
        """Print correspondence analysis statistics."""
        print(f"\nCorrespondence Statistics:")
        print(f"Average distance to nearest neighbor: {np.mean(distances[:, 0]):.4f}")
        print(f"Minimum distance: {np.min(distances[:, 0]):.4f}")
        print(f"Maximum distance: {np.max(distances[:, 0]):.4f}")
        print(f"Standard deviation: {np.std(distances[:, 0]):.4f}")

    def visualize_features(
        self,
        image: ImageArray,
        mask: Optional[ImageArray] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Features, Tuple[GridSize, TransformInfo]]:
        """
        Visualize features from a single image using PCA.
        
        Args:
            image: Input image
            mask: Optional mask for feature selection
            save_path: Optional path to save visualization
            
        Returns:
            Tuple of (features, grid info)
        """
        # Extract features
        _, grid_size, transform_info = self._prepare_image(image)
        features = self._extract_features(image)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(image)
        ax1.set_title("Original Image")
        
        # PCA visualization
        pca = PCA(n_components=3)
        if mask is not None:
            resized_mask = self._prepare_mask(mask, grid_size)
            masked_features = features[resized_mask]
            reduced_features = pca.fit_transform(masked_features.astype(np.float32))
            
            # Reconstruct full grid
            full_reduced = np.zeros((*resized_mask.shape, 3), dtype=reduced_features.dtype)
            full_reduced[resized_mask] = reduced_features
            reduced_features = full_reduced
        else:
            reduced_features = pca.fit_transform(features.astype(np.float32))
        
        # Reshape and normalize
        vis_features = reduced_features.reshape((*grid_size, -1))
        vis_features = (vis_features - vis_features.min()) / (vis_features.max() - vis_features.min())
        
        ax2.imshow(vis_features)
        ax2.set_title("CheXFound Feature Visualization")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return features, (grid_size, transform_info)

    def _prepare_mask(self, mask: ImageArray, grid_size: GridSize) -> np.ndarray:
        """Prepare mask to match token grid size."""
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        resized_mask = mask_image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        return np.asarray(resized_mask).flatten() > 127


def main() -> None:
    """Example usage demonstrating the refactored CheXFound matcher."""
    # Initialize matcher
    matcher = CheXFoundMatcher()
    
    # Load images
    print("Loading chest X-ray images...")
    image1 = np.array(Image.open('./Chest_Xray_PA_3-8-2010.png').convert('RGB'))
    image2 = np.array(Image.open('./Chest_Xray_2.jpg').convert('RGB'))
    
    # Example 1: Single image feature visualization
    print("\n=== Example 1: Feature Visualization ===")
    matcher.visualize_features(image1, save_path="outs/refactored_xray_visualization.png")
    
    # Example 2: Global correspondence analysis
    print("\n=== Example 2: Global Correspondence Analysis ===")
    matcher.analyze_global_correspondences(
        image1, image2, num_correspondences=20, save_prefix="outs/refactored_global"
    )
    
    # Example 3: Position-specific correspondences
    print("\n=== Example 3: Position-Specific Correspondences ===")
    query_positions = [(386, 1188)]
    
    correspondences, grid_info = matcher.find_correspondences(
        image1, image2, query_positions, k_neighbors=3
    )
    
    # Print results
    for i, corr in enumerate(correspondences):
        print(f"Query {i}: {corr.query_position}")
        for j, match in enumerate(corr.matches):
            print(f"  Match {j}: {match.position}, distance: {match.distance:.4f}")
    
    # Visualize correspondences
    matcher.visualize_correspondences(
        image1, image2, correspondences,
        title1="Image 1 (Query Points)", title2="Image 2 (Matched Points)",
        save_path="outs/refactored_position_correspondences.png"
    )
    
    # Example 4: Reverse direction (query from image2 to image1)
    print("\n=== Example 4: Reverse Direction Correspondences ===")
    reverse_query_positions = [(100, 402)]
    
    reverse_correspondences, reverse_grid_info = matcher.find_correspondences(
        image2, image1, reverse_query_positions, k_neighbors=2  # Note: swapped image order
    )
    
    # Visualize reverse correspondences
    matcher.visualize_correspondences(
        image2, image1, reverse_correspondences,
        title1="Image 2 (Query Points)", title2="Image 1 (Matched Points)",
        save_path="outs/reverse_position_correspondences.png"
    )
    
if __name__ == "__main__":
    main()
