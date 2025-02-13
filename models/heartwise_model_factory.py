class HeartWiseModelFactory:
    name = 'HeartWiseModelFactory'
    
    @classmethod
    def create_model(cls, model_config: dict):
        model_name = model_config['model_name']
        model_name_lower = model_name.lower()

        # Check if the current class matches the pipeline name
        if cls.name.lower() == model_name_lower:
            return cls(**(model_config))
                
        # Recursively check all subclasses
        for subclass in cls.__subclasses__():
            try:
                return subclass.create_model(model_config)
            except ValueError:
                continue  # Continue searching in other subclasses
        
        # If not found in any subclass, raise an error
        raise ValueError(f"Model '{model_name}' not found")