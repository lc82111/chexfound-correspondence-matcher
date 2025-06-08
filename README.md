# CheXFound Image Correspondence Matcher

This project provides functionality for finding correspondences between chest X-ray images using the CheXFound encoder.

## Features

- **Image Correspondence Matching**: Find corresponding regions between chest X-ray images
- **Feature Visualization**: Visualize extracted features using PCA
- **Global Correspondence Analysis**: Analyze correspondences across entire images
- **Position-Specific Matching**: Find matches for specific query positions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xchexfound.git
cd xchexfound
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download pretrained weights and configuration files:
   - Download the required files from: <https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806>
   - Create the following directory structure in your project:

   ```text
   /mnt/data/checkpoints/chexfound/
   ├── config.yaml
   └── teacher_checkpoint.pth
   ```

   - Place the downloaded `config.yaml` and `teacher_checkpoint.pth` files in the above directory
   - Alternatively, you can specify custom paths when initializing the `CheXFoundMatcher`:

   ```python
   matcher = CheXFoundMatcher(
       config_file="path/to/your/config.yaml",
       pretrained_weights="path/to/your/teacher_checkpoint.pth"
   )
   ```

## Usage

The following examples demonstrate the complete functionality of the CheXFoundMatcher using real chest X-ray images. You can run these examples by executing `python correspondency.py`.

### Complete Working Example

```python
from correspondency import CheXFoundMatcher
import numpy as np
from PIL import Image

# Initialize matcher
matcher = CheXFoundMatcher()

# Load chest X-ray images
print("Loading chest X-ray images...")
image1 = np.array(Image.open('./Chest_Xray_PA_3-8-2010.png').convert('RGB'))
image2 = np.array(Image.open('./Chest_Xray_2.jpg').convert('RGB'))
```

### Example 1: Feature Visualization

```python
# Visualize features from a single image using PCA
print("=== Example 1: Feature Visualization ===")
matcher.visualize_features(image1, save_path="outs/refactored_xray_visualization.png")
```

### Example 2: Global Correspondence Analysis

```python
# Analyze global correspondences between two images
print("=== Example 2: Global Correspondence Analysis ===")
matcher.analyze_global_correspondences(
    image1, image2, num_correspondences=20, save_prefix="outs/refactored_global"
)
```

### Example 3: Position-Specific Correspondences

```python
# Find correspondences for specific query positions
print("=== Example 3: Position-Specific Correspondences ===")
query_positions = [(386, 1188)]  # Specific anatomical point

correspondences, grid_info = matcher.find_correspondences(
    image1, image2, query_positions, k_neighbors=3
)

# Print results
for i, corr in enumerate(correspondences):
    print(f"Query {i}: {corr.query_position}")
    for j, match in enumerate(corr.matches):
        print(f"  Match {j}: {match.position}, distance: {match.distance:.4f}")

# Visualize correspondences with custom titles
matcher.visualize_correspondences(
    image1, image2, correspondences,
    title1="Image 1 (Query Points)", title2="Image 2 (Matched Points)",
    save_path="outs/refactored_position_correspondences.png"
)
```

### Example 4: Reverse Direction Correspondences

```python
# Find correspondences from second image to first image
print("=== Example 4: Reverse Direction Correspondences ===")
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
```

### Running the Complete Example

To run all examples at once, simply execute:

```bash
python correspondency.py
```

This will generate several output files in the `outs/` directory:
- `refactored_xray_visualization.png` - Feature visualization of the first image
- `refactored_global_distance_analysis.png` - Distance distribution analysis
- `refactored_global_visualization.png` - Global correspondence visualization
- `refactored_position_correspondences.png` - Position-specific correspondences
- `reverse_position_correspondences.png` - Reverse direction correspondences

## Project Structure

- `correspondency.py` - Main correspondence matching functionality
- `chexfound_encoder.py` - CheXFound encoder wrapper
- `chexfound/` - CheXFound model implementation
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.8+
- PyTorch
- PIL (Pillow)
- NumPy
- Matplotlib
- Scikit-learn
- Other dependencies listed in `requirements.txt`

## License

This project is for research purposes. Please ensure you have the appropriate model weights and follow the original CheXFound licensing terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
