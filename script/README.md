# Training Scripts

This directory contains training scripts for 4 different tasks, each corresponding to a dataset folder (1-4).

## Scripts

- `train_task1.py` - Train models on data from `datasets/1/`
- `train_task2.py` - Train models on data from `datasets/2/`
- `train_task3.py` - Train models on data from `datasets/3/`
- `train_task4.py` - Train models on data from `datasets/4/`

## Usage

### Basic Usage

Train CNN model for Task 1:
```bash
python script/train_task1.py --model cnn
```

Train TCN model for Task 2:
```bash
python script/train_task2.py --model tcn
```

### Parameters

All scripts support the following parameters:

- `--model`: Model type (`cnn`, `tcn`, or `cnn_transformer`, default: `cnn`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--validation_split`: Validation set ratio (default: 0.2)
- `--device`: Device (`cpu` or `cuda`, default: auto-select)

TCN-specific parameters:
- `--tcn_num_channels`: TCN channel list (default: `64 128 256`)
- `--tcn_kernel_size`: TCN kernel size (default: 3)
- `--tcn_dropout`: TCN dropout rate (default: 0.2)

CNN+Transformer-specific parameters:
- `--cnn_transformer_cnn_channels`: CNN channel list (default: `64 128 256`)
- `--cnn_transformer_d_model`: Transformer embedding dimension (default: 256)
- `--cnn_transformer_nhead`: Number of attention heads (default: 8)
- `--cnn_transformer_num_layers`: Number of transformer layers (default: 2)
- `--cnn_transformer_dim_feedforward`: Feedforward dimension (default: 512)
- `--cnn_transformer_dropout`: Dropout rate (default: 0.1)

### Examples

Train CNN model for Task 1 with custom parameters:
```bash
python script/train_task1.py --model cnn --epochs 200 --batch_size 64 --learning_rate 0.0005
```

Train TCN model for Task 3 with custom TCN parameters:
```bash
python script/train_task3.py --model tcn --tcn_num_channels 128 256 512 --tcn_kernel_size 5 --tcn_dropout 0.3
```

Train CNN+Transformer model for Task 1:
```bash
python script/train_task1.py --model cnn_transformer
```

Train CNN+Transformer model with custom parameters:
```bash
python script/train_task1.py --model cnn_transformer \
    --cnn_transformer_cnn_channels 128 256 512 \
    --cnn_transformer_d_model 512 \
    --cnn_transformer_nhead 16 \
    --cnn_transformer_num_layers 4
```

## Model Save Locations

Models are saved to the `models/` directory, organized by task ID:
- Task 1: `models/1/{model_name}/{folder_name}/` (e.g., `models/1/cnn/Glutamic Acid/`)
- Task 2: `models/2/{model_name}/{folder_name}/` (e.g., `models/2/cnn/Ratiometric with PRO & VAL/`)
- Task 3: `models/3/{model_name}/{folder_name}/` (e.g., `models/3/cnn/AA1/`)
- Task 4: `models/4/{model_name}/{folder_name}/` (e.g., `models/4/cnn/COG-N-519/`)

**Note:** All tasks train one model per folder, so each folder gets its own subdirectory.

Each model directory contains:
- `best_model.pth` - Best model during training
- `final_model.pth` - Final model after training
- `model_config.json` - Model configuration
- `class_names.pkl` - Class names
- `scaler.pkl` - Standardizer

## Data Structure

Each task folder (`datasets/1/`, `datasets/2/`, etc.) contains subfolders that represent different classes. The data loader automatically extracts class labels from the folder structure.

Example:
- `datasets/1/Glutamic Acid/file.csv` → Class: "Glutamic Acid"
- `datasets/2/Ratiometric with PRO & VAL/P0 V10/file.csv` → Class: "P0 V10"

