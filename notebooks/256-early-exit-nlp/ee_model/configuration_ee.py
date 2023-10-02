from transformers import PretrainedConfig, AutoConfig


class EarlyExitConfig(PretrainedConfig):
    model_type = "earlyexit"

    def __init__(
        self,
        base_model = "bert-base-uncased",
        **kwargs,
    ):
        self.base_model = base_model
        base_config = AutoConfig.from_pretrained(self.base_model)
        self.hidden_size = base_config.hidden_size
        self.num_hidden_layers = base_config.num_hidden_layers
        super().__init__(**kwargs)