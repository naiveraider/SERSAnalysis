# SERS Spectral Data Classification System

This is a deep learning-based spectral data classification system for classifying SERS (Surface-Enhanced Raman Spectroscopy) data. Implemented using PyTorch, it supports multiple model architectures.

## Features

- Automatically extracts spectral data from CSV files between `[SPECTRUM]` and `[ANALYSIS RESULT]` markers
- Supports multiple model architectures: CNN, TCN, etc.
- Supports multi-class classification
- Automatic data preprocessing and standardization
- Model training, validation, and saving
- GPU acceleration support
- Unified model factory system for easy extension of new models

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Data files should be in CSV format with the following structure:
- Spectral data starting from the `[SPECTRUM]` marker
- Data format: `wavelength;intensity`
- Ending at the `[ANALYSIS RESULT]` marker

Example:
```
[SPECTRUM]
100.00;-76.84
102.00;-46.36
...
[ANALYSIS RESULT]
```

## Dataset Organization

Datasets should be organized in the following structure:
```
datasets/
  ├── class1/
  │   ├── file1.csv
  │   └── file2.csv
  ├── class2/
  │   └── file1.csv
  └── ...
```

Folder names will be used as class labels.

## Usage

### 1. Training Models

#### Using CNN Model (default)
```bash
python -m src.train --model cnn --data_dir datasets --epochs 100 --batch_size 32
```

#### Using TCN Model
```bash
python -m src.train --model tcn --data_dir datasets --epochs 100 --batch_size 32
```

#### Using CNN+Transformer Model
```bash
python -m src.train --model cnn_transformer --data_dir datasets --epochs 100 --batch_size 32
```

#### TCN Model Specific Parameters
```bash
python -m src.train --model tcn \
    --tcn_num_channels 64 128 256 \
    --tcn_kernel_size 3 \
    --tcn_dropout 0.2 \
    --data_dir datasets
```

#### CNN+Transformer Model Specific Parameters
```bash
python -m src.train --model cnn_transformer \
    --cnn_transformer_cnn_channels 64 128 256 \
    --cnn_transformer_d_model 256 \
    --cnn_transformer_nhead 8 \
    --cnn_transformer_num_layers 2 \
    --data_dir datasets
```

**General Parameters:**
- `--model`: Model type, options: `cnn`, `tcn`, `cnn_transformer` (default: `cnn`)
- `--data_dir`: Dataset directory (default: datasets)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_save_path`: Model save path (default: model/saved_model)
- `--validation_split`: Validation set ratio (default: 0.2)
- `--device`: Device type, 'cpu' or 'cuda' (default: auto-select)

**TCN Model Specific Parameters:**
- `--tcn_num_channels`: List of channel numbers for each TCN layer (default: [64, 128, 256])
- `--tcn_kernel_size`: Convolution kernel size (default: 3)
- `--tcn_dropout`: Dropout ratio (default: 0.2)

**CNN+Transformer Model Specific Parameters:**
- `--cnn_transformer_cnn_channels`: List of channel numbers for each CNN layer (default: [64, 128, 256])
- `--cnn_transformer_d_model`: Transformer embedding dimension (default: 256)
- `--cnn_transformer_nhead`: Number of attention heads (default: 8)
- `--cnn_transformer_num_layers`: Number of transformer encoder layers (default: 2)
- `--cnn_transformer_dim_feedforward`: Feedforward network dimension (default: 512)
- `--cnn_transformer_dropout`: Dropout ratio (default: 0.1)

### 2. Using Models for Prediction

```bash
python -m src.predict --model_path model/saved_model --file_path datasets/1/Glutamic Acid/Glu 4-17 T001.csv
```

Parameters:
- `--model_path`: Model save path
- `--file_path`: Path to the CSV file to predict
- `--device`: Device type, 'cpu' or 'cuda' (default: auto-select)

## Supported Model Architectures

### CNN Model
Standard 1D Convolutional Neural Network, including:
- 3 1D convolutional layers (64, 128, 256 filters)
- Batch normalization layers
- Max pooling layers
- Dropout layers (to prevent overfitting)
- Global average pooling layer
- 2 fully connected layers (512, 256 neurons)
- Softmax output layer

### TCN Model
Temporal Convolutional Network, including:
- Multi-layer causal convolution
- Dilated convolution to capture long-term dependencies
- Residual connections
- Weight normalization
- Global average pooling layer
- Fully connected layer
- Softmax output layer

The TCN model is particularly suitable for processing sequential data and can capture long-term dependencies.

### CNN+Transformer Model
CNN + Transformer hybrid model that combines the advantages of both architectures:
- **CNN Component**: Extracts local features
  - Multi-layer 1D convolution
  - Batch normalization
  - Max pooling
- **Transformer Component**: Captures global dependencies
  - Positional encoding
  - Multi-head self-attention mechanism
  - Feedforward network
  - Layer normalization
- **Feature Fusion**: Combines CNN and Transformer features
- **Classification Head**: Fully connected layers for classification

The CNN+Transformer model combines CNN's local feature extraction capability with Transformer's global dependency modeling capability, making it suitable for processing complex spectral data.

**Note:** All models automatically use GPU (if available), otherwise CPU.

## File Structure

```
SERSAnalysis/
├── src/                    # Source code directory
│   ├── __init__.py
│   ├── data_loader.py      # Data loader
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   └── model/              # Model definitions
│       ├── __init__.py
│       ├── cnn_model.py       # CNN model definition
│       ├── tcn_model.py        # TCN model definition
│       ├── cnn_transformer_model.py  # CNN+Transformer model definition
│       └── model_factory.py    # Model factory (unified management of all models)
├── script/                 # Training scripts for different tasks
│   ├── train_task1.py
│   ├── train_task2.py
│   ├── train_task3.py
│   └── train_task4.py
├── datasets/               # Dataset directory
├── models/                 # Saved models directory (generated after training)
│   ├── 1/                  # Task 1 models
│   │   └── {model_name}/   # e.g., cnn/, tcn/, cnn_transformer/
│   │       └── {folder_name}/  # For Task 1, each folder has its own model
│   ├── 2/                  # Task 2 models
│   ├── 3/                  # Task 3 models
│   └── 4/                  # Task 4 models
│       └── {model_name}/   # Each model directory contains:
│           ├── best_model.pth      # Best model
│           ├── final_model.pth     # Final model
│           ├── model_config.json   # Model configuration
│           ├── class_names.pkl     # Class names
│           └── scaler.pkl          # Standardizer
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── example_train.sh       # Example training script
```

## Model Selection Recommendations

- **CNN Model**: Suitable for most cases, fast training speed, fewer parameters
- **TCN Model**: Suitable for sequential data that requires capturing long-term dependencies, usually requires more training time
- **CNN+Transformer Model**: Suitable for complex spectral data, combines local and global features, usually has the best performance but requires longer training time

## Notes

1. Ensure sufficient data for training (recommend at least 10-20 samples per class)
2. If data volume is small, you may need to adjust model parameters or use data augmentation
3. Training process automatically saves the best model and final model
4. Models use early stopping and learning rate decay to optimize training
5. If the system has an NVIDIA GPU with CUDA installed, models will automatically use GPU acceleration for training
6. Model files are saved in `.pth` format (PyTorch standard format)
7. Model configuration is automatically saved, and prediction will automatically identify model type
8. New model architectures can be easily added through the model factory system
