class HeartWiseModelFactory:
    """Factory to instantiate ECG models by name (BERT, WCR, EfficientNet, etc.)."""

    name = 'HeartWiseModelFactory'

    @classmethod
    def create_model(cls, model_config: dict):
        """Return a model instance for the given model_config; model_config must include 'model_name' and device/key args."""
        model_name = model_config['model_name']
        model_name_lower = model_name.lower()

        # Find the class before constructing it: loader ValueErrors must reach the caller.
        candidates = [cls]
        while candidates:
            candidate = candidates.pop()
            if candidate.name.lower() == model_name_lower:
                return candidate(**model_config)
            candidates.extend(reversed(candidate.__subclasses__()))
        raise ValueError(f"Model '{model_name}' not found")
