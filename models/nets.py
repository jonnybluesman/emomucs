
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


activations = nn.ModuleDict([
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()],
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['selu', nn.SELU()],
                ['elu', nn.ELU()]
])


def compute_flattened_maps(cnn_layers, input_shape):
    """
    Utility function to compute the size of the flattened feature maps
    after the convolutional layers, which should be passed as input
    together with the shape of the input tensors.
    """
    
    x = torch.randn(1, 1, *input_shape)
    with torch.no_grad():
        x = cnn_layers(x)
    
    return np.prod(x.shape[1:])

        
def cnn_weights_init(m):
    """
    Reinitialise the parameters of a network with custom init functions.
    Xavier initialisation is  used for Linear layers, whereas convolutional
    layers are initialised with the Hu Kaiming method for more stability.
    
    This method only supports, for the moment, conv and linear layers. The idea
    of this method is "reset and refine", which ensures that all layer are reinit.
    """
    if ("reset_parameters" in dir(m)):
        m.reset_parameters() # first of all reset the layer
    
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        
        
def torch_weights_init(m):
    """
    Reinitialise the parameters of a layer as the good torch would do.
    This method is not very useful as it is right now.
    """
    if ("reset_parameters" in dir(m)):
        m.reset_parameters() # first reset the layer
    # TODO: do something more from the specs


def create_dense_block(
    in_feats, out_feats,  architecture=['fc', 'act', 'drop'],
    activation='relu', dropout_prob=0, wrapping=True):
    """
    Factory method for fully connected layers, with the possibility
    to choose the activation function and the regularisation technique.

    TODO:
        - add the support for batch normalisation;
    """
    assert all(name in ['fc', 'act', 'drop'] for name in architecture)
    
    dense_block = {
        'fc': nn.Linear(in_feats, out_feats),
        'act' : activations[activation],
        'drop': nn.Dropout(p=dropout_prob),
    }
    
    dense_block = [dense_block[name] for name in architecture]
    return nn.Sequential(*dense_block) if wrapping else dense_block


def create_2d_convolutional_block(
    in_feats, num_filters, filter_size, architecture=['bn', 'act', 'pool', 'drop'],
    pool_size=(2,2), padding=0, stride=1, activation='relu', dropout_prob=0):
    """
    Factory method for convolutional layers, with the possibility.
    
    Args:
        in_features (int): number of input features;
        num_filters (int): number of kernels;
        filter_size (tuple): size of the 2D filters/kernels;
        architecture (list): list of strings describing the cnn items;
        pool_size (tuple): size of the pooling operation (same of stride);
        padding (int or tuple): the amount of padding for each dimension;
        stride (int or tuple): stride of the convolutional kernel;
        activation (str): namne of the activation function;
        dropout_prob (float): probability of dropping out;
    
    """
    
    assert all(name in ['bn', 'act', 'pool', 'drop'] for name in architecture)
    
    cnn_block = {
        'bn'  : nn.BatchNorm2d(num_filters),
        'act' : activations[activation],
        'pool': nn.MaxPool2d(pool_size),
        'drop': nn.Dropout(p=dropout_prob),
    }
    
    return nn.Sequential(
        nn.Conv2d(in_feats, num_filters, filter_size, 
            padding=padding, stride=stride),
        *[cnn_block[name] for name in architecture])


class DeezerConv1d(nn.Module):
    """
    Simple implementation of the AudioCNN presented in 
    "Music Mood Detection Based On Audio And Lyrics With Deep Neural Net".
    
    Code adapted from https://github.com/Dohppak/Music_Emotion_Recognition
    """
    
    def __init__(self, input_shape, n_kernels=[32, 16], kernel_sizes=[8, 8],
                 mpool_stride=[4, 4], fc_units=[64, 2]):
        """
        Class constructor for the creation of a static 1DCNN.
        
        Args:
            input_shape (2-tuple): (number of mel bands, frames).
            n_kernels (2-tuple): number of 1D filters per conv layer;
            kernel_sizes (2-tuple): size of kernels as number of frames;
            mpool_stride (2-tuple): strides of 1D max pooling (same as size);
            fc_units (2-tuple): number of units in the last fully-connected layers.
            
        TODO:
            - Class constructor from sample input instead of specifying nmel;
            - The parameterisation of the net can be more beautiful;
            - It is still not clear which activation function is used in the first FCL.
        """
        super(DeezerConv1d, self).__init__()
        
        self.flattened_size = int(np.floor(
            ((np.floor((input_shape[1] - kernel_sizes[0] + 1) / mpool_stride[0])) 
             - kernel_sizes[1] + 1) / mpool_stride[1]) * n_kernels[-1])
        
        self.conv_blocks = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(input_shape[0], n_kernels[0], kernel_size=kernel_sizes[0]),
                nn.MaxPool1d(mpool_stride[0], stride=mpool_stride[0]),
                nn.BatchNorm1d(n_kernels[0])),
            nn.Sequential(
                nn.Conv1d(n_kernels[0], n_kernels[1], kernel_size=kernel_sizes[1]),
                nn.MaxPool1d(mpool_stride[1], stride=mpool_stride[1]),
                nn.BatchNorm1d(n_kernels[1]))
        )

        self._fcl = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=self.flattened_size, out_features=fc_units[0]),
            #nn.Tanh(),  # we use a relu instead
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=fc_units[0], out_features=fc_units[1]),
        )

        self.apply(self._init_weights)
    
    
    def convolutional_features(self, x):
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)
        
    
    def forward(self, x):
        x_cnn_flat = self.convolutional_features(x)
        pred = self._fcl(x_cnn_flat)
        return pred


    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)

        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


class VGGishEmoNet(nn.Module):
    """
    A VGG-based 2dCNN typically used for music tagging and transfer learning as in:
    "Transfer learning for music classification and regression tasks"
    
    Architecture inspired from https://github.com/keunwoochoi/transfer_learning_music/
    """
    
    def __init__(
        self, input_shape, n_kernels=[32]*5, kernel_sizes=[(3,3)]*5,
        pooling_sizes=None, dropout=0., cnn_activation='elu', fc_units=2):
        """
        Class constructor for the creation of a static 2DCNN.
        
        Args:
            input_shape (2-tuple): (number of mel bands, frames).
            n_kernels (list): number of 2D filters per conv layer;
            kernel_sizes (list): size of kernels for each conc layer;
            pooling_sizes (list): size of each 2D maxpooling operation;
            dropout (float): probability of dropping out conv activations;
            cnn_activation (str): name of the activation function for conv layers;
            fc_units (int): number of units in the last fully-connected layer.
            
        TODO:
            - The parameterisation of the net can be more beautiful;
        """
        super(VGGishEmoNet, self).__init__()
        
        if pooling_sizes is None:
            pooling_sizes = get_vggish_poolings_from_features(*input_shape)
        assert len(n_kernels) == len(kernel_sizes) == len(pooling_sizes)
        
        conv_input_shapes = [1] + n_kernels[:-1]
        cnn_arch = ['bn', 'act', 'pool', 'drop']
        conv_blocks = [create_2d_convolutional_block(
            conv_input_shape, n_kernel, kernel_size, cnn_arch,
            pooling_size, 1, 1, cnn_activation, dropout) \
            for conv_input_shape, n_kernel, kernel_size, pooling_size \
                in zip(conv_input_shapes, n_kernels, kernel_sizes, pooling_sizes)]
        
        self.conv_blocks = nn.Sequential(
            *conv_blocks, nn.AdaptiveAvgPool2d((1, 1)))
        # the following operation is not needed as we already have the adaptive pooling
        # self.flattened_size = compute_flattened_maps(self.conv_blocks, input_shape)
        self.flattened_size = n_kernels[-1]
        
        self._fcl = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=fc_units),
        )
    

    def convolutional_features(self, x):
        x = x.unsqueeze(1) # to ensure n_channels is 1
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)

    
    def forward(self, x):
        x_cnn_flat = self.convolutional_features(x)
        pred = self._fcl(x_cnn_flat)
        return pred
    
    
class VGGishExplainable(nn.Module):
    """
    A VGG-based 2dCNN designed for explainable MER, presented in:
    "Towards explainable MER, by using mid-level features".
    This is the model that is denoted as A2E in the paper.
    """
    
    def __init__(
        self, input_shape, n_kernels=[64, 64, 128, 128, 256, 256, 384, 512, 256],
        kernel_sizes=[(5,5)]+[(3,3)]*8, pooling_sizes=[(2, 2), (2, 2)],
        strides=[2]+[1]*8, paddings=[2]+[1]*7+[0], dropout=[.3, .3], 
        cnn_activation='relu', fc_units=2):
        """
        Class constructor for the creation of a static 2DCNN.
        
        Args:
            input_shape (2-tuple): (number of mel bands, frames).
            n_kernels (list): number of 2D filters per conv layer;
            kernel_sizes (list): size of kernels for each conc layer;
            pooling_sizes (list): size of each 2D maxpooling operation;
            dropout (float): probability of dropping out conv activations;
            cnn_activation (str): name of the activation function for conv layers;
            fc_units (int): number of units in the last fully-connected layer.
            
        TODO:
            - The parameterisation of the net can be more beautiful;
        """
        super(VGGishExplainable, self).__init__()
        assert len(n_kernels) == len(kernel_sizes) == len(strides) == len(paddings)
        
        conv_input_shapes = [1] + n_kernels[:-1]
        conv_blocks = [create_2d_convolutional_block(
            conv_input_shape, n_kernel, kernel_size, ['bn', 'act'],
            None, padding, stride, cnn_activation) \
            for conv_input_shape, n_kernel, kernel_size, padding, stride \
                in zip(conv_input_shapes, n_kernels, kernel_sizes, paddings, strides)]
        
        self.conv_blocks = nn.Sequential(
            *conv_blocks[:2],
            nn.MaxPool2d(pooling_sizes[0]),
            nn.Dropout(p=dropout[0]),
            *conv_blocks[2:4],
            nn.MaxPool2d(pooling_sizes[1]),
            nn.Dropout(p=dropout[1]),
            *conv_blocks[4:],
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # the following operation is not needed as we already have the adaptive pooling
        # flattened_size = compute_flattened_maps(self.conv_blocks, input_shape)
        self.flattened_size = n_kernels[-1]
        
        self._fcl = nn.Sequential(
            nn.Linear(in_features=self.flattened_size, out_features=fc_units),
        )


    def convolutional_features(self, x):
        x = x.unsqueeze(1) # to ensure n_channels is 1
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)

    
    def forward(self, x):
        x_cnn_flat = self.convolutional_features(x)
        pred = self._fcl(x_cnn_flat)
        return pred

    
def get_vggish_poolings_from_features(n_mels=96, n_frames=1360):
    """
    Get the pooling sizes for the standard VGG-based model for audio tagging.
    Code from: https://github.com/keunwoochoi/transfer_learning_music/blob/master/models_transfer.py

    Todo:
        - This code is ugly, reorganise in a config file;
        - This method is assuming (at the moment) a certain number of frames (1360 covering 30s);
    """

    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 72:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 64:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 48:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 32:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (2, 4)]
    elif n_mels >= 24:
        poolings = [(2, 4), (2, 4), (2, 5), (3, 4), (1, 4)]
    elif n_mels >= 18:
        poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (3, 4)]
    elif n_mels >= 18:
        poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (3, 4)]
    elif n_mels >= 16:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (1, 4)]
    elif n_mels >= 12:
        poolings = [(2, 4), (1, 4), (2, 5), (3, 4), (1, 4)]
    elif n_mels >= 8:
        poolings = [(2, 4), (1, 4), (2, 5), (2, 4), (1, 4)]
    elif n_mels >= 6:
        poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (1, 4)]
    elif n_mels >= 4:
        poolings = [(2, 4), (1, 4), (2, 5), (1, 4), (1, 4)]
    elif n_mels >= 2:
        poolings = [(2, 4), (1, 4), (1, 5), (1, 4), (1, 4)]
    else:  # n_mels == 1
        poolings = [(1, 4), (1, 4), (1, 5), (1, 4), (1, 4)]
    
    ratio = n_frames / 1360  # as these measures are referred to this unit
    # print([(poo_w, pool_l * ratio) for poo_w, pool_l in poolings])
    return [(poo_w, round(pool_l * ratio)) for poo_w, pool_l in poolings]


def simple_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_shape_flows_through(conv_blocks, feat_shape):
    """
    Currently works just for a CNN...
    
    TODO:
        - Make it general for any network
    """
    
    print('Generating random batch of 2 x {} data'.format(feat_shape))
    x = torch.rand((2, *feat_shape))

    conv_blocks.eval()
    conv_blocks.to(device=torch.device('cpu'), dtype=torch.float)
    x.to(device=torch.device('cpu'), dtype=torch.float)

    print("Initial shape: {}".format(x.shape))
    for i, layer in enumerate(conv_blocks):

        if isinstance(layer, nn.Sequential):
            for j, sub_layer in enumerate(layer):

                x = sub_layer(x)
                if isinstance(sub_layer, nn.Conv2d) or isinstance(sub_layer, nn.MaxPool2d):
                    print("Layer {} ({}) | Shape after {}: {} "
                          .format(i, j, sub_layer.__class__.__name__, x.shape))

        else:
            x = layer(x)
            # only print if the level is expected to afffect the shape
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
                print("Layer {} | Shape after {}: {} "
                          .format(i, layer.__class__.__name__, x.shape))