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

### Basic Usage

```python
from correspondency import CheXFoundMatcher
import numpy as np
from PIL import Image

# Initialize matcher
matcher = CheXFoundMatcher()

# Load images
image1 = np.array(Image.open('image1.png').convert('RGB'))
image2 = np.array(Image.open('image2.png').convert('RGB'))

# Find correspondences for specific positions
query_positions = [(100, 200), (300, 400)]
correspondences, grid_info = matcher.find_correspondences(
    image1, image2, query_positions, k_neighbors=3
)

# Visualize results
matcher.visualize_correspondences(
    image1, image2, correspondences,
    save_path="correspondences.png"
)
```

### Feature Visualization

```python
# Visualize features from a single image
matcher.visualize_features(image1, save_path="features.png")
```

### Global Correspondence Analysis

```python
# Analyze global correspondences between two images
distances, indices = matcher.analyze_global_correspondences(
    image1, image2, num_correspondences=20
)
```

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
