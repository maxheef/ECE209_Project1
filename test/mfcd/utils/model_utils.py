from typing import Optional, Dict, Any


class ModelInfo:
    def __init__(
            self,
            model_class,
            model_config: Dict[str, Any] = None,
            processor_config: Optional[Dict[str, Any]] = None,
            generate_config: Optional[Dict[str, Any]] = None,
            data_preparation_config: Optional[Dict[str, Any]] = None,
            special_model_config_for_dataset: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.model_class = model_class
        self.model_config = model_config if model_config is not None else {}
        self.processor_config = processor_config if processor_config is not None else {}
        self.generation_config = generate_config if generate_config is not None else {}
        self.data_preparation_config = data_preparation_config if data_preparation_config is not None else {}
        self.special_model_config_for_dataset = (
            special_model_config_for_dataset
            if special_model_config_for_dataset is not None
            else {}
        )