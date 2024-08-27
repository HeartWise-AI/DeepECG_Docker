




class ModelFactory:
    @classmethod
    def create_model(cls, model_config):
        model_name = model_config['model_name']
        model_name_lower = model_name.lower()
        for subclass in cls.__subclasses__():
            if subclass.name.lower() == model_name_lower:
                return subclass(**model_config)
        raise ValueError(f"Model {model_name} not found")




