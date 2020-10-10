
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

from nets import DeezerConv1d, VGGishEmoNet, get_vggish_poolings_from_features
from nets import create_dense_block, create_2d_convolutional_block
from data import SOURCE_NAMES

logger = logging.getLogger("emomucs")


class Emomucs(nn.Module):
    """
    The Emomucs network, a demucs-based model for MER.
    """
    
    def __init__(self, source_names, source_model, input_shape, fusion_method, 
                 dropout_probs=[.3,.3], prediction_units=[32, 2], finetuning=True, **kwargs):
        """
        Class constructor for the creation of an Emoucs net.
        
        Args:
            source_names (list): list of source names to use (see `data.SOURCE_NAMES`);
            source_model (Module): the class corresponfing to the source model to instantiate.
            input_shape (2-tuple): (number of mel bands, frames).
            fusion_method (str): either `early` or `late` at the moment.
            finetuning (bool): true if the source models need to be trained.
            
        TODO:
            - The parameterisation of the net can be more beautiful;
            - source model can be a list of model types, and the kwargs
                should also contain a list of hparams, one for each model.
            - 
            - the fine-tuning can be individual per source.
        
        """
        assert all([s_name in SOURCE_NAMES for s_name in source_names])
        prediction_units = [units for units in prediction_units if units > 0]
        assert len(dropout_probs) == len(prediction_units)
        assert fusion_method in ["early", "mid", "late"]
        super(Emomucs, self).__init__()
        
        self.source_names = source_names
        self.source_models = nn.ModuleList()
        self.fusion_method = fusion_method
        
        for source_name in source_names: # creating each source-model
            self.source_models.append(source_model(input_shape, **kwargs))
        
        if not finetuning:
            logger.info("Fine-tuning is off: keeping source models frozen.")
            for source_model in self.source_models:
                for param in source_model.parameters():
                    param.requires_grad = False
            
        if fusion_method == "early":
            # Case 1: we concatenate the reshaped feature maps after conv layers
            num_combined = sum(
                [source_model.flattened_size for source_model in self.source_models])
        elif fusion_method == "mid":
            # Case 2: concatenation before the regression level (only for deezeremo)
            num_combined = sum(
                [source_model._fcl[1].out_features for source_model in self.source_models])
        else:
            # Case 3: we use the full output of the models (last layer)
            num_combined = sum(
                [source_model._fcl[-1].out_features for source_model in self.source_models])
        
        
        units_per_fcl, dense_bocks = [num_combined] + prediction_units, []
        config_per_fcl = list(zip(units_per_fcl[:-1], units_per_fcl[1:], dropout_probs))
        for i, (in_units, out_units, dropb) in enumerate(config_per_fcl):
            
            architecture = ['drop', 'fc', 'act'] \
                if i < len(dropout_probs) - 1 else ['drop', 'fc']
            dense_bocks.extend(create_dense_block(
                in_units, out_units, architecture=architecture, 
                dropout_prob=dropb, wrapping=False))
        
        self._fcl = nn.Sequential(*dense_bocks)
    
    
    def forward(self, x):
        
        ## Check if the number of streams corresponds
        # assert x.shape[1] == len(self.source_name)

        # Forward each source to the right network
        models_out = []
        
        for i, source_model in enumerate(self.source_models):
            if self.fusion_method == "early":
                models_out.append(source_model.convolutional_features(x[:, i]))
            elif self.fusion_method == "mid":
                models_out.append(source_model._fcl[:3](
                    source_model.convolutional_features(x[:, i])))
            else:
                models_out.append(source_model(x[:, i]))
                
        models_out = torch.cat(models_out, 1)
        
        return self._fcl(models_out)

    
#     def to(self, *args, **kwargs):
#         """ 
#         Moves and/or casts the parameters and buffers. This ensures that
#         all the source models are relocated to the right device.
#         """
#         # run the default torch method first
#         super().to(*args, **kwargs)
#         # then apply it recursively for each submodule
#         for source_model in self.source_models:
#             source_model.to(*args, **kwargs)

    
    def get_source_model(self, source_name):
        """
        Return the model associated to the given source name, if present.
        
        Args:
            source_name (str): name of a source.
        """
        assert source_name in self.source_names
        return self.source_models[self.source_names.index(source_name)]

    
    def _update_source_model(self, source_name, state_dict):
        """
        Update the state dict of a source model with the provided one.
        """
        logger.info(f"Updating state dict of the {source_name} model.")
        self.get_source_model(source_name).load_state_dict(state_dict)
        
    
    def update_source_models(self, state_dict_per_model):
        """
        Update the state dict of all the source models from the specifications
        contained in the `state_dict_per_model` mapping. The mapping can be from
        source_name to either file path to the .pt file with the state_dict,
        or the state dict by itself (already loaded).
        """
        assert all(source_name in self.source_names for \
                   source_name in state_dict_per_model.keys())
        
        logger.info(f"Updating the state dict of {state_dict_per_model.keys()}")
        for source_name, source_sd_spec in state_dict_per_model.items():
            
            state_dict = torch.load(source_sd_spec) \
                if isinstance(source_sd_spec, str) else source_sd_spec
            self._update_source_model(source_name, state_dict)

            
def warmup_emomucs(model, model_name, checkpoint_dir, device=None, fold=None):
    """
    Warm-up function for Emomucs: loading the source models
    
    Args:
        model (nn.Module): an instance of the Emomucs model to pre-process before Nested CV;
        model_name (str): name of the emomucs model, to retrieve the type of the submodels.
        checkpoint_dir (str): where the checkpoints of the source models will be retrieved;
        fold (2-tuple): the outer and the inner fold (e.g. 0, 1 for out fold 0 and inner 1).
    """
    device = torch.device("cpu") if device is None else device
    fold_substr = f"_{fold[0]}_{fold[1]}" if fold is not None else ""
    
    submodel_name = model_name.split("_")[1]
    
    source_models_sd_paths = {
        source_name: torch.load(
            os.path.join(checkpoint_dir, 
                         f"{submodel_name}_{source_name}{fold_substr}.pt"), 
            map_location=device)
        for source_name in model.source_names}
    
    
    model.update_source_models(source_models_sd_paths)
    

    
class EmomucsUnified(nn.Module):
    """
    The Emomucs network, a demucs-based model for MER, with aggregated sources.
    """
    
    def __init__(self, source_names, input_shape, 
                 n_kernels=[32]*5, kernel_sizes=[(3,3)]*5, pooling_sizes=None,
                 cnn_dropout=0, cnn_activation='elu',
                 dropout_probs=[.3,.3], prediction_units=[32, 2], act_fn='relu'):
        """
        Classs constructor for the Emomucs with aggregated sources.
        
        Args:
            source_names (list of str): the source streams to process;
            input_shape (tuple): number of mel-bands x number of frames;
            n_kernels (list of int): number of convolutional kernels per layer;
            kernel_size (list of tuples): shape of each 2D convolutional kernel;
            pooling_sizes (list of tuples): shape of the 2D maxpooling operator;
            cnn_dropout (float): the droput probability for the convolutional layers;
            cnn_activation (str): the activation function for the convolutional layers;
            dropout_probs (list of float): dropout probabilities for the fully-conn blocks;
            prediction_units (list of int): number of units for each fully-conn block.
        
        TODO:
            - Understand the suitability of poolings (if they can be the same of vggemonet).
            - The number of filters should be proportional to the number of sources/channels.
        """
        assert all([s_name in SOURCE_NAMES for s_name in source_names])
        assert len(dropout_probs) == len(prediction_units)
        super(EmomucsUnified, self).__init__()
        
        self.source_names = source_names
        
        if pooling_sizes is None:
            pooling_sizes = get_vggish_poolings_from_features(*input_shape)
        assert len(n_kernels) == len(kernel_sizes) == len(pooling_sizes)
        
        conv_input_shapes = [len(source_names)] + n_kernels[:-1]
        cnn_arch = ['bn', 'act', 'pool', 'drop']
        conv_blocks = [create_2d_convolutional_block(
            conv_input_shape, n_kernel, kernel_size, cnn_arch,
            pooling_size, 1, 1, cnn_activation, cnn_dropout) \
            for conv_input_shape, n_kernel, kernel_size, pooling_size \
                in zip(conv_input_shapes, n_kernels, kernel_sizes, pooling_sizes)]
        
        self.conv_blocks = nn.Sequential(
            *conv_blocks, nn.AdaptiveAvgPool2d((1, 1)))
        # the following operation is not needed as we already have the adaptive pooling
        # self.flattened_size = compute_flattened_maps(self.conv_blocks, input_shape)
        self.flattened_size = n_kernels[-1]
        
        units_per_fcl, dense_bocks = [n_kernels[-1]] + prediction_units, []
        config_per_fcl = list(zip(units_per_fcl[:-1], units_per_fcl[1:], dropout_probs))
        for i, (in_units, out_units, dropb) in enumerate(config_per_fcl):
            
            architecture = ['drop', 'fc', 'act'] \
                if i < len(dropout_probs) - 1 else ['drop', 'fc']
            dense_bocks.extend(create_dense_block(
                in_units, out_units, architecture=architecture, 
                dropout_prob=dropb, wrapping=False, activation=act_fn))
        
        self._fcl = nn.Sequential(*dense_bocks)

        
        
    def forward(self, x):
        x_cnn_fmap = self.conv_blocks(x)
        x_cnn_fmap = x_cnn_fmap.view(x.size(0), -1)
        return self._fcl(x_cnn_fmap)
        
    