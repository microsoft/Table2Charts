from data import DataConfig, TokenType, AnaType, Segment
from typing import Optional


class ModelConfig:
    """Hyper-parameters for the Input Embedding part"""

    def __init__(self, data_config: DataConfig,
                 hidden_size: int = 128, dropout: float = 0.1, position_sensitive: bool = True):
        num_token_type = len(TokenType) if data_config.unified_ana_token else len(TokenType) + len(AnaType) - 1
        num_segment_type = len(Segment)

        self.embed_len = data_config.embed_len
        self.data_len = data_config.data_len
        self.cat_len = len(data_config.cat_nums)
        self.positional = position_sensitive
        self.num_token_type = num_token_type
        self.num_segment_type = num_segment_type
        self.num_categories = data_config.cat_nums

        self.hidden = hidden_size
        self.dropout = dropout

class CopyNetConfig(ModelConfig):
    """Hyper-parameters for the CopyNet architecture"""

    def __init__(self, data_config: DataConfig,
                 encoder_layers: int = 2, encoder_GRU_hidden: int = 256, encoder_hidden_size: int = 128,
                 decoder_layers: int = 1, decoder_GRU_hidden: int = 160, decoder_hidden_size: int = 80,
                 dropout: float = 0.1, position_sensitive: bool = False):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        # Command tokens: [SEP], GroupingOperations and AggFunctions
        self.num_cmd_tokens = data_config.num_cmd_tokens()
        self.encoder_GRU_hidden = encoder_GRU_hidden
        self.decoder_GRU_hidden = decoder_GRU_hidden
        self.feature_dim = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        super().__init__(data_config, hidden_size=encoder_hidden_size, dropout=dropout,
                         position_sensitive=position_sensitive)

    def __str__(self):
        return "{}el{}fd{}.{}GRUh".format(self.encoder_layers, self.feature_dim, self.encoder_GRU_hidden,
                                          self.decoder_GRU_hidden)


DEFAULT_MODEL_SIZES = ["small", "medium", "large", "super", "resize_small", "shallow_large",
                       "shallower_large", "shallowest_large", "customize"]
DEFAULT_MODEL_NAMES = ["cp"]


def small_cp_config(data_config: DataConfig, position_sensitive: bool = False):
    return CopyNetConfig(
        data_config,
        encoder_hidden_size=192,
        encoder_layers=2,
        encoder_GRU_hidden=128,
        decoder_hidden_size=192,
        decoder_layers=1,
        decoder_GRU_hidden=128,
        dropout=0.1,
        position_sensitive=position_sensitive
    )


def medium_cp_config(data_config: DataConfig, position_sensitive: bool = False):
    return CopyNetConfig(
        data_config,
        encoder_hidden_size=320,
        encoder_layers=2,
        encoder_GRU_hidden=192,
        decoder_hidden_size=256,
        decoder_layers=1,
        decoder_GRU_hidden=192,
        dropout=0.1,
        position_sensitive=position_sensitive
    )


def large_cp_config(data_config: DataConfig, position_sensitive: bool = False):
    return CopyNetConfig(
        data_config,
        encoder_hidden_size=512,
        encoder_layers=2,
        encoder_GRU_hidden=288,
        decoder_hidden_size=512,
        decoder_layers=1,
        decoder_GRU_hidden=320,
        dropout=0.1,
        position_sensitive=position_sensitive
    )


def super_cp_config(data_config: DataConfig, position_sensitive: bool = False):
    return CopyNetConfig(
        data_config,
        encoder_hidden_size=256,
        encoder_layers=2,
        encoder_GRU_hidden=256,
        dropout=0.1,
        position_sensitive=position_sensitive
    )


def get_cp_config(data_config: DataConfig, size: str):
    if size == "small":
        return small_cp_config(data_config)
    elif size == "medium":
        return medium_cp_config(data_config)
    elif size == "large":
        return large_cp_config(data_config)
    elif size == "super":
        return super_cp_config(data_config)
    else:
        raise NotImplementedError(f"CopyNet config for {size} not yet implemented.")
