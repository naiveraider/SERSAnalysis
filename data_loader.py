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
        
        # Try multiple encodings to handle different file formats
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_content = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        # If all encodings failed, try with error handling
        if file_content is None:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.readlines()
        
        for line in file_content:
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
    
    def load_data_from_folder(self, folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all CSV files from a single folder
        
        Args:
            folder_path: Path to the folder containing CSV files
            
        Returns:
            X: Spectrum intensity data (n_samples, n_features)
            y: Labels (n_samples,) - all samples from same folder have same label (folder name)
        """
        X_list = []
        folder_name = folder_path.name
        
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Load all CSV files from the folder
        csv_files = list(folder_path.glob('*.csv'))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in folder: {folder_path}")
        
        for file_path in csv_files:
            spectrum = self.load_spectrum_from_file(file_path)
            if len(spectrum) > 0:
                # Only use intensity values (second column) as features
                intensities = spectrum[:, 1]
                X_list.append(intensities)
        
        if len(X_list) == 0:
            raise ValueError(f"No valid spectrum data found in folder: {folder_path}")
        
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
        # All samples from same folder - use folder name as label
        y = np.array([folder_name] * len(X))
        
        return X, y
    
    def prepare_data_from_folder(self, folder_path: Path, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare training and test data from a single folder
        
        Args:
            folder_path: Path to the folder containing CSV files
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test, class_name
        """
        X, y = self.load_data_from_folder(folder_path)
        
        # Since all samples are from the same folder, we have a single class
        # But we still need to encode for consistency
        y_encoded = np.zeros(len(y))  # All samples belong to class 0 (single class)
        class_name = folder_path.name
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train and test sets
        # For single class, we can't use stratify, but we still split
        if len(X) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=test_size, 
                random_state=random_state
            )
        else:
            # If only one sample, use it for training
            X_train, X_test = X_scaled, X_scaled[:0]
            y_train, y_test = y_encoded, y_encoded[:0]
        
        # Reshape data to CNN input format (samples, channels, length)
        # PyTorch expects (samples, channels, length), so (samples, 1, length)
        if len(X_train) > 0:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        if len(X_test) > 0:
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train, X_test, y_train, y_test, [class_name]
