
import configparser

from nets import DeezerConv1d, VGGishEmoNet, VGGishExplainable
from nets import get_vggish_poolings_from_features, cnn_weights_init, torch_weights_init

# TODO: default parameters go here ...

NAMES_TO_MODELS = {
    "deezeremo": DeezerConv1d,
    "vggemonet": VGGishEmoNet,
    "vggexp": VGGishExplainable
}


def hparams_from_config(config_path):
    """
    Read a config file with the hparameters configuration,
    and return them, after parsing, as a dictionary.
    
    TODO:
        - Sanity checks on types and values.
    """

    config = configparser.ConfigParser()

    config.read(config_path)
    hparams_cfg = config['HPARAMS']

    hparams_dict = {
        'mse_reduction': hparams_cfg.get('mse_reduction', 'sum'),
        'num_epochs': hparams_cfg.getint('num_epochs', 10000),
        'patience': hparams_cfg.getint('patience', 20),
        'lr': hparams_cfg.getfloat('lr', 10000),
        'batch_sizes': [int(b_size) for b_size in hparams_cfg.get(
            'batch_sizes', '32, 32, 32').split(',')]
    }
    
    return hparams_dict


def get_model_from_selection(model_name, input_shape):
    """
    Returns the model instantiated from the specification,
    together with the function to reinitialise the weights.
    
    We only support 3 models at the moment:
        i.e. DeezerConv1d, VGGishEmoNet, VGGishExplainable
        
    Args:
        model_name (str): one of 'deezeremo', 'vggemonet', 'vggexp';
        input_shape (tuple): shape of the input tensors.
    """
    
    reinit_fn = torch_weights_init
    
    if model_name == 'deezeremo':
        sel_model = DeezerConv1d(input_shape)
    
    elif model_name == 'vggemonet':
        sel_model = VGGishEmoNet(input_shape)
    
    elif model_name == 'vggexp':
        sel_model = VGGishExplainable(input_shape)
        
    else:
        raise ValueError("deezeremo, vggemonet, vggexp are supported!")
        
    return sel_model, reinit_fn
        
        
    
    