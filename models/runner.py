"""
Custom runner for the MER experiment.
"""

import os
import sys
import joblib
import random
import logging
import argparse

import torch
import numpy as np
import pandas as pd

from emomucs import Emomucs, EmomucsUnified, warmup_emomucs
from nets import cnn_weights_init, torch_weights_init
from data import StaticDataset, MultiSourceStaticDataset, SOURCE_NAMES
from experiment import NestedCrossValidation
from config import hparams_from_config, get_model_from_selection, NAMES_TO_MODELS
from utils import is_file, create_dir, str_list

from os.path import dirname, abspath
prj_abp = dirname(abspath(__file__))


def prepare_emomucs_experiment(args):
    
    dataset = MultiSourceStaticDataset(
        features_fname=args.audio_features,
        annotations_fname=args.va_annotations,
        source_names=args.sources
    )
    
    if args.model == "unified":
        # Single unified emomucs architecture for all sources.
        sel_model = EmomucsUnified(
            source_names=args.sources,
            input_shape=dataset[0][0].shape[1:],
            n_kernels=[20*len(args.sources)]*5,
            cnn_dropout=0.3,
            dropout_probs=args.dropouts,
            prediction_units=[args.units_per_source*len(args.sources), 2],
        )
        model_name = "emomucs_unified_" + '-'.join(args.sources)
        
    else:
        # Emomucs with source-specific models following the baselines.
        sel_model = Emomucs(
            source_names=args.sources,
            source_model=NAMES_TO_MODELS[args.model], 
            input_shape=dataset[0][0].shape[1:],
            fusion_method=args.fusion,
            finetuning=args.finetuning,
            dropout_probs=args.dropouts,
            prediction_units=[args.units_per_source*len(args.sources), 2], 
        )
        model_name = f"emomucs_{args.model}_{args.fusion}_{args.training}" \
            + f"_finet-{args.finetuning}_dp-{str_list(args.dropouts, True)}" \
            + f"_fcns-{str_list([args.units_per_source*len(args.sources), 2])}" \
            + "_"+ '-'.join(args.sources)
    
    return dataset, sel_model, model_name


def prepare_baseline_experiment(args):
    
    if args.sources is not None:
        assert len(args.sources) == 1, \
            "baseline mode does not support multi-source input"
        dataset = MultiSourceStaticDataset(
            features_fname=args.audio_features,
            annotations_fname=args.va_annotations,
            source_names=args.sources,
            transform=lambda x, y: (x[0], y)
         )
        model_name = f"{args.model}_{args.sources[0]}"

    else:
        dataset = StaticDataset(
            features_fname=args.audio_features,
            annotations_fname=args.va_annotations
        )
        model_name = args.model

    sel_model, _ = get_model_from_selection(
        args.model, dataset[0][0].shape
    )
    
    return dataset, sel_model, model_name


def run_nested_cross_validation_exp(args):
    """
    Prepare data and run nested CV ...
    """

    if args.mode == "emomucs":
        dataset, sel_model, model_name = prepare_emomucs_experiment(args)
        warmup_fn = warmup_emomucs \
            if args.training == 'load' and args.model != "unified" else None
    else:  # baseline run, use default params and no warmup function
        dataset, sel_model, model_name = prepare_baseline_experiment(args)
        warmup_fn = None
    
    print(sel_model)
    
    # from config file to hparamas dictionary
    hparams = hparams_from_config(args.config)
    hparams['num_epochs'] = 1 if args.debug else hparams['num_epochs']
    model_name = f"{model_name}_debug" if args.debug else model_name
    
    ncv_exp = NestedCrossValidation(
        sel_model, dataset, args.nestedcv_folds, hparams, 
        args.dtype, args.checkpointing, model_name, args.checkpoint_dir)
    ncv_exp.run_nested_cv_evaluation(args.folds, args.device, warmup_fn)
    
    if args.write:
        pd.DataFrame(ncv_exp.get_outer_fold_test_scores()).T.to_csv(
            os.path.join(args.result_dir, f"{model_name}_evaluation.csv"))
        pd.DataFrame(ncv_exp.get_targets_and_predictions()).to_csv(
            os.path.join(args.prediction_dir, f"{model_name}_annotations.csv"), index=False)


def activate_logging(log_dir):
    """
    Setup logging routines.
    """
    create_dir(log_dir)
    print('Logging in: {}\n'.format(log_dir))
    logger = logging.getLogger('emomucs')
    logger.setLevel(logging.DEBUG)
    # Creating a std out handler for the root logger
    chandler = logging.StreamHandler(sys.stdout)
    chandler.setLevel(logging.INFO)
    # Formatting style and adding the handler to the logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    chandler.setFormatter(formatter)
    logger.addHandler(chandler)


def main():
    """
    Main function to parse the arguments and call the main process.
    
    TODO:
        - add support for output target selectin (from names);
        - replace the default config by embedding it in the config module.
    """
    
    parser = argparse.ArgumentParser(
        description='Models for MER | Evaluation with nested cross-validation.')
    
    
    parser.add_argument('mode', choices=['baseline', 'emomucs'],
                        help='Either running the baselines or emomucs.')
    parser.add_argument('model', choices=['deezeremo', 'vggemonet', 'vggexp', 'unified'],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('audio_features', type=lambda x: is_file(parser, x),
                        help='Path to the file containing the audio features of the dataset.')
    parser.add_argument('va_annotations', type=lambda x: is_file(parser, x),
                        help='Path to the csv file with the Valence-Arousal annotations.')
    parser.add_argument('nestedcv_folds', type=lambda x: is_file(parser, x),
                        help='Path to the file containing the dataset split for nested cv.')
    
    # The following arguments are specifically for using Emomucs
    parser.add_argument('--sources',  action='store', nargs='+',
                        help=f'One or more source names from {SOURCE_NAMES} for Emomucs')
    parser.add_argument('--fusion', action='store', type=str,
                        choices=['early', 'mid', 'late'], default="early",
                        help='Feature fusion technique to use in Emomucs.')
    parser.add_argument('--finetuning', action='store_true', default=False,
                        help='Whether fine-tuning each source model already trained separately.')
    parser.add_argument('--training', action='store', choices=['joint', 'load'], default='load',
                        help='Train the source-models from scratch in emomucs or load them.')
    parser.add_argument('--dropouts', action='store', nargs='+',
                        help='Dropout probs for the fully-connected layers of Emomucs.')
    parser.add_argument('--units_per_source', type=int, action='store', default=32,
                        help='Number of fully-connected units per source in Emomucs.')
    
    parser.add_argument('--log_dir', action='store',
                        help='Directory where log files will be generated.')
    parser.add_argument('--result_dir', action='store',
                        help='Where the evaluation results will be saved.')
    parser.add_argument('--checkpoint_dir', action='store',
                        help='Where the model checkpoints will be saved.')
    parser.add_argument('--write', action='store_true', default=False,
                        help='Whether to write the results to disk or not.')
    parser.add_argument('--checkpointing', action='store_true', default=False,
                        help='Whether the model state dict will be saved at every fold.')
    parser.add_argument('--config', action='store', type=str,
                        default=os.path.join(prj_abp, 'example_config.ini'),
                        help='File containing the specification of the hyperparameters.')
    
    parser.add_argument('--sel_output_features', action='store', nargs='+',
                        help='Optional list of the output features names for regression.')
    parser.add_argument('--folds', action='store', nargs='+',
                        help='List of outer folds to process (for parallel execution mode.)')
    parser.add_argument('--num_workers', action='store', type=int, default=0,
                        help='Number of workers for data loading.')
    parser.add_argument('--dtype', action='store', choices=['d', 'f'], default='f',
                        help='Data type of tensors to process. Default: f (float).')
    parser.add_argument('--device', action='store',
                        help='Device to use for training and validation. Default: cpu.')
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed for the reproducibility of the experiment.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Whether to activate the debug mode for exp checking.')

    args = parser.parse_args()
    
    # TODO: check the consistency of the paramaters ...
    assert args.sources is None or all([name in SOURCE_NAMES for name in args.sources])
    args.dropouts = [.5, .5] if args.dropouts is None \
        else [float(drop) for drop in args.dropouts[:2]]
    args.folds = [int(fold_no) for fold_no in args.folds] if args.folds is not None else None
    
    # Filling missing/optional values if not provided ...
    args.dtype = torch.double if args.dtype == 'd' else torch.float
    args.device = torch.device('cpu') if args.device is None else torch.device(args.device)

    # setting the random seed for all modules
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        # torch.cuda.set_device(args.device)
    
    if args.log_dir is None:
        args.log_dir = os.path.join(
            os.path.dirname(args.va_annotations), 'logdir')
    activate_logging(args.log_dir)
    
    args.result_dir = create_dir(os.path.join(args.log_dir, "results_ft")) \
        if args.result_dir is None else create_dir(args.result_dir)
    args.checkpoint_dir = create_dir(os.path.join(args.log_dir, "checkpoints")) \
        if args.checkpoint_dir is None else create_dir(args.checkpoint_dir)
    args.prediction_dir = create_dir(os.path.join(args.result_dir, "predictions"))
    
    print('Using {} (tensor type: {}) with random seed {} |'
          ' Running fold(s): {} and logging in: {}'
          .format(args.device, args.dtype, args.seed,
                  'all' if args.folds is None else args.folds, args.log_dir))
    
    run_nested_cross_validation_exp(args)


if __name__ == "__main__":
    main()
