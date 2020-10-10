
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def compute_squared_errors(model_annotations, col_suffix="_mean"):
    """Simple utility for analysing the predictions of a model."""
    
    if isinstance(model_annotations, str):
        model_annotations = pd.read_csv(source_anns_path, index_col=False)
    # replacing the mean suffix if present
    model_annotations.columns = [cname.replace(col_suffix, "") 
                                 for cname in model_annotations.columns]
    var_names = set([name.split("_")[1] for name in model_annotations.columns])
    
    # Step 1: compute the SE for each source model
    se_dict = {"se_" + name : np.power(
        model_annotations["target_" + name].values 
        - model_annotations["out_" + name].values, 2) for name in var_names}
    # Step 2: if more than 1 prections are available, then compute the avg
    if len(se_dict) > 1:  # compute the average of the SEs if possible
        se_dict.update({"se_mean": np.mean(list(se_dict.values()), axis=0)})
        
    return se_dict


def compute_rsquared(model_annotations, col_suffix="_mean"):
    """Simple utility for analysing the predictions of a model."""
    
    if isinstance(model_annotations, str):
        model_annotations = pd.read_csv(source_anns_path, index_col=False)
    # replacing the mean suffix if present
    model_annotations.columns = [cname.replace(col_suffix, "") 
                                 for cname in model_annotations.columns]
    var_names = set([name.split("_")[1] for name in model_annotations.columns])
            
    # Step 1: compute the R2 for each source model
    r2_dict = {"r2_" + name : r2_score(
        model_annotations["target_" + name].values, 
        model_annotations["out_" + name].values) for name in var_names}
    # Step 2: if more than 1 prections are available, then compute the avg
    if len(r2_dict) > 1:  # compute the average of the SEs if possible
        r2_dict.update({"r2_mean": np.mean(list(r2_dict.values()), axis=0)})
    
    return r2_dict


def compute_rmse_from_sq_errors(sq_errors_df):
    """
    Simple utility to compute the evaluation measures.
    """
    return dict(np.sqrt(np.mean(sq_errors_df)))


def evaluate_prediction(annotation_df):
    """
    Returns the squared errors and the evaluation of the annotations.
    """
    sq_errors_df = pd.DataFrame(compute_squared_errors(annotation_df))
    
    eval_dict = compute_rmse_from_sq_errors(sq_errors_df)
    eval_dict.update(compute_rsquared(annotation_df))
    
    return sq_errors_df, eval_dict

