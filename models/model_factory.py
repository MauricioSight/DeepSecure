from models.seqwatch import SeqWatch

class ModelFactory:
    def __init__(self, config):
        self.config = config

        self.model_config = config.get('model')
        self.model_name = self.model_config.get('name')

    def create_model(self):
        if self.model_name == "SeqWatch":
            feature_config = self.config.get('feature')
            w_size = feature_config.get('window_size')
            number_of_bytes = feature_config.get('number_of_bytes')

            return SeqWatch(w_size=w_size, input_size=number_of_bytes)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
