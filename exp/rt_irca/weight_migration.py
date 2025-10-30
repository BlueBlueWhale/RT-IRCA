"""
RT-IRCA model initialization, weight migration and verification

This module provides functionality for migrating weights from YOLO11 models to RT-IRCA models.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from ultralytics import YOLO


@dataclass
class ModelConfig:
    """Model configuration data class."""
    model_name: str
    yaml: str
    pretrained_model: str
    mapped_model: str


# Model configuration constants
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'n': ModelConfig(
        model_name='RT-IRCA-n',
        yaml='rtirca_n.yaml',
        pretrained_model='yolo11n.pt',
        mapped_model='rtirca_n.pt',
    ),
    'l': ModelConfig(
        model_name='RT-IRCA-l',
        yaml='rtirca_l.yaml',
        pretrained_model='yolo11l.pt',
        mapped_model='rtirca_l.pt',
    )
}

# Constants for model processing
INSERTION_POINT = 10  # IRCA module insertion position in the model architecture
TOLERANCE = 1e-6  # Tolerance for weight comparison during verification


class RTIRCAWeightMigration:
    """RT-IRCA weight migration class."""
    
    def __init__(self, model_type: str):
        """
        Initialize RT-IRCA weight migration.
        
        Args:
            model_type: Model type identifier ('n' for nano, 'l' for large)
            
        Raises:
            ValueError: If model_type is not one of the supported types in MODEL_CONFIGS
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(MODEL_CONFIGS.keys())}")
        
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self._yolo11_model: Optional[YOLO] = None
        self._rtirca_model: Optional[YOLO] = None
    
    @property
    def yolo11_model(self) -> YOLO:
        """Lazy loading of YOLO11 model."""
        if self._yolo11_model is None:
            print(f"Loading YOLO11 model: {self.config.pretrained_model}")
            self._yolo11_model = YOLO(self.config.pretrained_model)
        return self._yolo11_model
    
    @property
    def rtirca_model(self) -> YOLO:
        """Lazy loading of RT-IRCA model."""
        if self._rtirca_model is None:
            print(f"Loading RT-IRCA model: {self.config.mapped_model}")
            self._rtirca_model = YOLO(self.config.mapped_model)
        return self._rtirca_model
    
    def _get_layer_index(self, param_name: str) -> Optional[int]:
        """
        Extract layer index from parameter name.
        
        Args:
            param_name: Parameter name in the format 'X.param_name' where X is the layer index
            
        Returns:
            Optional[int]: Layer index if found, None otherwise
        """
        parts = param_name.split('.')
        if parts and parts[0].isdigit():
            return int(parts[0])
        return None
    
    def _map_parameter_name(self, param_name: str, layer_idx: int) -> str:
        """
        Map parameter name based on layer index.
        
        Args:
            param_name: Original parameter name from the source model
            layer_idx: Layer index extracted from the parameter name
            
        Returns:
            str: Mapped parameter name for the target model architecture
        """
        if layer_idx <= INSERTION_POINT:
            return param_name
        else:
            new_layer_idx = layer_idx + 1
            return param_name.replace(f"{layer_idx}.", f"{new_layer_idx}.", 1)
    
    def _print_verification_results(self, total_param_groups: int, successfully_mapped: int) -> None:
        """
        Print verification result statistics.
        
        Args:
            total_param_groups: Total number of parameter groups in the model
            successfully_mapped: Number of parameter groups successfully mapped
        """
        print(f"  Total parameter groups: {total_param_groups}")
        print(f"  Successfully mapped parameter groups: {successfully_mapped}")
    
    def verify_model_weights(self) -> Tuple[int, int]:
        """
        Verify if model weights are correctly migrated.
        
        Returns:
            Tuple[int, int]: (Total parameter groups, Successfully mapped parameter groups)
        """
        yolo11_state_dict = self.yolo11_model.model.model.state_dict()
        rtirca_state_dict = self.rtirca_model.model.model.state_dict()
        
        print("\n=== Weight Migration Verification ===")
        
        # Statistics
        print("\nWeight Migration Statistics:")
        total_param_groups = 0
        successfully_mapped = 0
        
        for name, param_tensor in yolo11_state_dict.items():
            layer_idx = self._get_layer_index(name)
            if layer_idx is None:
                continue
                
            total_param_groups += 1
            target_name = self._map_parameter_name(name, layer_idx)
            
            if (target_name in rtirca_state_dict and 
                rtirca_state_dict[target_name].shape == param_tensor.shape and
                torch.allclose(param_tensor, rtirca_state_dict[target_name], atol=TOLERANCE)):
                successfully_mapped += 1
        
        self._print_verification_results(total_param_groups, successfully_mapped)
        print("\n=== Verification Completed ===\n")
        return total_param_groups, successfully_mapped
    
    def _create_weight_mapping(self, yolo11_state_dict: Dict, rtirca_state_dict: Dict) -> Dict:
        """
        Create weight mapping dictionary.
        
        Args:
            yolo11_state_dict: State dictionary of the YOLO11 model
            rtirca_state_dict: State dictionary of the RT-IRCA model
            
        Returns:
            Dict: Mapping dictionary with RT-IRCA parameter names as keys and
                 YOLO11 parameter tensors as values
        """
        new_state_dict = {}
        
        for name, param_tensor in yolo11_state_dict.items():
            layer_idx = self._get_layer_index(name)
            if layer_idx is None:
                continue
                
            target_name = self._map_parameter_name(name, layer_idx)
            
            # Check if parameter exists and shape matches
            if (target_name in rtirca_state_dict and 
                rtirca_state_dict[target_name].shape == param_tensor.shape):
                new_state_dict[target_name] = param_tensor
        
        return new_state_dict
    
    def init_rtirca_model(self) -> YOLO:
        """
        Initialize RT-IRCA model and map weights.
        
        Returns:
            YOLO: Initialized RT-IRCA model with mapped weights
        """
        print(f"Initializing {self.config.model_name} model...")
        
        # Initialize RT-IRCA model architecture
        rtirca_model = YOLO(self.config.yaml)
        rtirca_state_dict = rtirca_model.model.model.state_dict()
        
        # Get YOLO11 model weights
        yolo11_state_dict = self.yolo11_model.model.model.state_dict()
        
        # Create weight mapping dictionary
        new_state_dict = self._create_weight_mapping(yolo11_state_dict, rtirca_state_dict)
        
        # Load mapped weights
        rtirca_model.model.model.load_state_dict(new_state_dict, strict=False)
        
        # Save initialized model
        rtirca_model.save(self.config.mapped_model)
        print(f"{self.config.model_name} model initialized and weights mapped successfully!")
        
        return rtirca_model


def _print_batch_summary(results: Dict[str, Tuple[int, int]]) -> None:
    """
    Print batch processing summary results.
    
    Args:
        results: Dictionary with model type as key and (total, success) tuple as value
    """
    print(f"\n{'='*50}")
    print("Batch Processing Summary")
    print(f"{'='*50}")
    
    for model_type, (total, success) in results.items():
        print(f"{model_type}: {success}/{total}")


def batch_init_and_verify(model_types: List[str] = None) -> Dict[str, Tuple[int, int]]:
    """
    Batch initialize and verify multiple models.
    
    Args:
        model_types: List of model types to process. If None, processes all available models.
                    
    Returns:
        Dict[str, Tuple[int, int]]: Dictionary with model type as key and a tuple containing:
            - Total parameter groups in the model
            - Number of parameter groups successfully mapped
    """
    if model_types is None:
        model_types = list(MODEL_CONFIGS.keys())
    
    results = {}
    
    for model_type in model_types:
        try:
            print(f"\n{'='*50}")
            print(f"Processing model type: {model_type}")
            print(f"{'='*50}")
            
            # Initialize model
            migration = RTIRCAWeightMigration(model_type)
            model = migration.init_rtirca_model()
            
            # Verify weights
            total, success = migration.verify_model_weights()
            results[model_type] = (total, success)
            
        except Exception as e:
            print(f"Error processing model {model_type}: {str(e)}")
            results[model_type] = (0, 0)
    
    _print_batch_summary(results)
    return results


if __name__ == "__main__":
    # Batch process all models
    batch_init_and_verify(['n', 'l'])