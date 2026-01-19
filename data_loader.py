"""
Data Loader: Extract data between [SPECTRUM] and [ANALYSIS RESULT] from CSV files
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SpectrumDataLoader:
    """Spectrum data loader"""
    
    def __init__(self, data_dir: str = "datasets", task_id: int = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Dataset directory path
            task_id: Task ID (1-4), if specified, only load data from datasets/{task_id}/
        """
        self.data_dir = Path(data_dir)
        self.task_id = task_id
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_spectrum_from_file(self, file_path: Path) -> np.ndarray:
        """
        Extract spectrum data from a single CSV file
        
        Args:
            file_path: CSV file path
            
        Returns:
            Spectrum data array (wavelength, intensity)
        """
        spectrum_data = []
        in_spectrum_section = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Detect [SPECTRUM] marker
                if line == '[SPECTRUM]':
                    in_spectrum_section = True
                    continue
                
                # Detect [ANALYSIS RESULT] marker, stop reading
                if line == '[ANALYSIS RESULT]':
                    break
                
                # If in spectrum section, parse data
                if in_spectrum_section and line:
                    try:
                        parts = line.split(';')
                        if len(parts) == 2:
                            wavelength = float(parts[0])
                            intensity = float(parts[1])
                            spectrum_data.append([wavelength, intensity])
                    except ValueError:
                        continue
        
        return np.array(spectrum_data)
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all dataset
        
        Returns:
            X: Spectrum intensity data (n_samples, n_features)
            y: Labels (n_samples,)
            labels: Class name list
        """
        X_list = []
        y_list = []
        label_names = []
        
        # Determine the base directory to search
        if self.task_id is not None:
            search_dir = self.data_dir / str(self.task_id)
            if not search_dir.exists():
                raise ValueError(f"Task directory not found: {search_dir}")
        else:
            search_dir = self.data_dir
        
        # Traverse dataset directory
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = Path(root) / file
                    
                    # Extract class label from file path
                    # For datasets/1/Glutamic Acid/file.csv -> Glutamic Acid
                    # For datasets/2/Ratiometric with PRO & VAL/P0 V10/file.csv -> P0 V10
                    # Find the immediate subfolder of the task folder
                    parts = file_path.parts
                    
                    # Find task folder index
                    task_idx = None
                    for i, part in enumerate(parts):
                        if part == str(self.task_id) if self.task_id else part in ['1', '2', '3', '4']:
                            task_idx = i
                            break
                    
                    if task_idx is not None and len(parts) > task_idx + 1:
                        # Get the first subfolder after task folder as class label
                        label = parts[task_idx + 1]
                    elif len(parts) >= 2:
                        # Fallback: use parent folder name
                        label = parts[-2]
                    else:
                        label = "Unknown"
                    
                    # Load spectrum data
                    spectrum = self.load_spectrum_from_file(file_path)
                    
                    if len(spectrum) > 0:
                        # Only use intensity values (second column) as features
                        intensities = spectrum[:, 1]
                        X_list.append(intensities)
                        y_list.append(label)
                        label_names.append(label)
        
        if len(X_list) == 0:
            raise ValueError("No valid spectrum data files found")
        
        # Convert to numpy array
        # Find maximum length and pad shorter sequences
        max_len = max(len(x) for x in X_list)
        X_padded = []
        for x in X_list:
            if len(x) < max_len:
                # Zero padding
                x_padded = np.pad(x, (0, max_len - len(x)), mode='constant')
            else:
                x_padded = x
            X_padded.append(x_padded)
        
        X = np.array(X_padded)
        y = np.array(y_list)
        
        return X, y, label_names
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare training and test data
        
        Args:
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test, class_names
        """
        X, y, _ = self.load_all_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_names = self.label_encoder.classes_
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Reshape data to CNN input format (samples, channels, length)
        # PyTorch expects (samples, channels, length), so (samples, 1, length)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train, X_test, y_train, y_test, class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        _, y, _ = self.load_all_data()
        return len(np.unique(y))
