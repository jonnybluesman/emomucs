
import os
import re
import glob
import math

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

from training import train_network, RegressionModelEvaluator
from data import load_nested_cv_fold, create_data_loaders
from nets import cnn_weights_init, torch_weights_init
from utils import create_dir


def nn_train_evaluate_model(
    model, tr_loader, va_loader, hparameters, dtype, device, loggers=None, logdir=None):
    """
    The given model is then trained and evaluated on the validation set after each
    epoch, so as to implement the early stropping strategy.

    Args:
        model (torch.nn.Module): the model to train and evaluate;
        tr_loader (DataLoader): data loader for the training set;
        va_loader (DataLoader): data loader for the validation set;
        hparameters (dict): a dictionary with the training hyper-parameters.
        ...
        
    ***TODO: Run the models multiple times with different weight initialisation.***

    Returns the trained model and the history of the training/validation loss.
    """

    # setting loggers verbosity for the training/evaluation step
    loggers = [False] * 3 if loggers is None else [False, True, True]
    (net_trained, min_loss), hist = train_network(
        model, tr_loader, va_loader, hparameters=hparameters,
        dtype=dtype, device=device, loggers=loggers, log_dir=logdir
    )

    return net_trained, min_loss, hist


class NestedCrossValidation(object):
    """
    Nested cross-validation experiment for the evaluation of a regression model.
    
    Args:
        model (nn.Module): the model to evaluate in the nested CV loop;
        dataset (data.Dataset): the full dataset that will be splitted in folds;
        fold_path (str): path to the file with the nested cv splits;
        hparams (dict): hyper-parameters as a mapping from hparam name to value; 
        dtype (torch dtype): the type of tensors to use for data and parameters;
        model_name (str): name of the model (for checkpointing purposes);
        checkpoint_dir (str): where to save fold checkpoints. None if not needed.
        
    TODO:
        - create fold dict file if null is specified.
        
    """

    def __init__(self, model, dataset, fold_path, hparams, dtype,
                 checkpointing=False, model_name=None, checkpoint_dir=None):
                            
        self.model = model
        self.dtype = dtype
        self.hparams = hparams
        self.dataset = dataset
        
        self.checkpointing = checkpointing
        self.model_name = model.__class__.__name__ \
            if model_name is None else model_name
        self.checkpoint_dir = create_dir(checkpoint_dir)
        
        self.ncv_dict = load_nested_cv_fold(fold_path)
        self.test_losses = dict() # {out_fold: test losses}
        self.best_models = dict() # {out_fold: best model}
        self.targets, self.predictions = [], []
    
    
    def get_outer_fold_test_scores(self):
        """
        Simple getter for the results of the Nested CV experiment.
        
        TODO:
            - Warning in case not all the outer fold were run.
        """
        
        assert len(self.test_losses) > 0, "Nested CV not started yet!"
        return self.test_losses
    
    
    def get_targets_and_predictions(self):
        """
        Simple getter for the results of the Nested CV experiment.
        
        TODO:
            - Warning in case not all the outer fold were run.
        """
        
        assert len(self.targets) > 0, "Nested CV not started yet!"
        all_targets = torch.cat(self.targets, 0).numpy()
        all_predics = torch.cat(self.predictions, 0).numpy()
        out_names = self.dataset.annotation_df.columns
        
        return dict(
            **{"out_" + out_names[i] : all_predics[:, i] 
                   for i in range(len(out_names))},
            **{"target_" + out_names[i] : all_targets[:, i] 
                   for i in range(len(out_names))})
        
    
    def get_outer_fold_best_models(self, sel_folds=None):
        """
        Simple getter for the results of the Nested CV experiment.
        
        TODO:
            - Warning in case not all the outer fold were run.
            - This should not work, as it is not a deep copy.
        """
        assert len(self.best_models)> 0, "Nested CV not started yet!"
        sel_folds = self.best_models if sel_folds is None else sel_folds
        assert all([fold in self.best_models.keys() for fold in sel_folds])
        
        return {fold: model for fold, model in self.best_models.items() if fold in sel_folds}
    
    
    def run_nested_cv_evaluation(
        self, sel_folds, device, warmup_fn=None, reinit_fn=torch_weights_init):
        """
        Perform Nested CV on the selected outer folds using the provided device.
        Designed to be adapted for parallelisation.

        Args:
            sel_folds (list): list of outer fold identifiers to run;
            device (torch device): which device to use for experimentation;
            warmup_fn (fucntion): optional preprocessing function for the model; 
            reinit_fn (function): how to reinit the net, `torch_weights_init` as default.
        
        TODO:
            - Run on all outer folds if sel_folds is None;
            - Split function in process_outer_fold and process_inner_fold; 

        """
        warmup_fn = warmup_fn if warmup_fn is not None \
            else lambda model, model_name, logdir, dev, fold : model
        
        sel_folds = list(self.ncv_dict.keys()) if sel_folds is None else sel_folds
        assert all([fold not in self.test_losses.keys() for fold in sel_folds]), \
            "One or more of the specified folds has/have already been run!"
        batch_sizes = self.hparams.get("batch_sizes")
        ncv_dict_sel = {outer_fold : outer_fold_data for outer_fold, outer_fold_data
                    in self.ncv_dict.items() if outer_fold in sel_folds}

        for outer_fold, outer_fold_data in ncv_dict_sel.items():

            test_ids = outer_fold_data.pop('test_ids')    
            te_loader = DataLoader(
                Subset(self.dataset, test_ids), shuffle=False,
                batch_size=len(test_ids) if batch_sizes[-1] is None else batch_sizes[-1])

            print("Outer fold {} starting | {} test samples".format(outer_fold, len(test_ids)))

            innerfold_models = dict()  # dictionary {validation loss: network}
            for inner_fold, inner_fold_data in outer_fold_data.items():
                
                warmup_fn(self.model, self.model_name, self.checkpoint_dir, device, (outer_fold, inner_fold))

                tr_ids, va_ids = inner_fold_data['training_ids'], inner_fold_data['validation_ids']
                tr_loader, va_loader = create_data_loaders(self.dataset, tr_ids, va_ids)

                print('... processing fold {}-{} --- training samples: {}, validation samples: {}'
                          .format(outer_fold, inner_fold, len(tr_ids), len(va_ids)))

                tr_loader, va_loader = create_data_loaders(
                    self.dataset, tr_ids, va_ids, batch_sizes=batch_sizes[:2], num_workers=0
                )

                infold_model, infold_validloss, _ = nn_train_evaluate_model(
                    self.model, tr_loader, va_loader, self.hparams,
                    dtype=self.dtype, device=device, loggers=[False, True, True]
                )

                innerfold_models[infold_validloss] = infold_model
                
                if self.checkpointing:
                    torch.save(infold_model.state_dict(), os.path.join(
                        self.checkpoint_dir, f"{self.model_name}_{outer_fold}_{inner_fold}.pt"))
                
                self.model.apply(reinit_fn)  # reset model params for next fold
                

            infold_best = innerfold_models[min(innerfold_models.keys())]
            self.best_models[outer_fold] = infold_best  # or just save the state dict
            
            te_evaluator = RegressionModelEvaluator(infold_best, device, self.dtype)
            rmse, r2score, targets, predictions = te_evaluator.evaluate_net_raw(te_loader)
            out_names = self.dataset.annotation_df.columns
            
            self.targets.append(targets)
            self.predictions.append(predictions)
            self.test_losses[outer_fold] = {
                **dict(zip(['rmse_' + col for col in out_names], rmse)),
                **dict(zip(['r2score_' + col for col in out_names], r2score))
            }
            
