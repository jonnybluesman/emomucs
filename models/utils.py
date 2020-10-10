
import matplotlib.pyplot as plt
import pandas as pd

import os
import joblib


def plot_training_history(training_history):

    pd.DataFrame(training_history).plot()
    
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Loss history during training')

    plt.show()

    
def is_file(parser, f_arg):
    if not os.path.exists(f_arg):
        return parser.error("File %s does not exist!" % f_arg)
    return f_arg


def create_dir(path):
    """Create dir if it does not exist."""
    if (path is not None) and (not os.path.exists(path)):
        os.mkdir(path)
    return path


def save_data(dicted_data, save_path, compression=0):
    """
    Simple utility to save dicted data in joblib.
    """
        
    assert os.path.exists(os.path.dirname(save_path))    

    with open(save_path, "wb") as f:
        joblib.dump(dicted_data, f, compress=compression)
        
        
def str_list(a_list, keep_decimal=False, separator="-"):
    """
    Convert a list to a single string.
    """
    s_list = [str(item) for item in a_list]
    s_list = [fitem.split(".")[1] for fitem in s_list] \
        if keep_decimal else s_list
    return '-'.join(s_list)