from models.seqwatch import SeqWatch
from models.tcn_anomly_detector import TNCAnomalyDetector

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
        
        if self.model_name == "TNCAnomalyDetector":
            feature_config = self.config.get('feature')
            number_of_bytes = feature_config.get('number_of_bytes')
            
            hidden_size = self.model_config.get('hidden_size')
            levels = self.model_config.get('levels')
            kernel_size = self.model_config.get('kernel_size')
            T = self.model_config.get('T')
            dropout = self.model_config.get('dropout')

            return TNCAnomalyDetector(input_size=number_of_bytes, hidden_size=hidden_size, levels=levels, 
                                      kernel_size=kernel_size,  T=T, dropout=dropout)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
