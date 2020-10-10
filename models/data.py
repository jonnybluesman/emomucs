
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, random_split

SOURCE_NAMES = ["drums", "bass", "other", "vocals"]


class StaticDataset(Dataset):
    """
    Static dataset for summative music emotion recognition in the Valence-Arousal space.
    """

    def __init__(self, features_fname, annotations_fname, track_id="track_id", transform=None):
        """
        Args:
            features_fname (str): path to the audio features compressed file (in joblib).
            annotations_fname (str): path to the csv file with the summative annotations.
            transform (callable, optional): optional transform to be applied on a sample.

        TODO:
            - also check whether the annotation track_ids are more than the song ids.
            - consider the possibility of applying the rescaler here;
        """

        self.annotation_df = pd.read_csv(annotations_fname, index_col=track_id)
        self.track_ids = list(self.annotation_df.index)
        self.annotations = self.annotation_df.values
        
        with open(features_fname, "rb") as f:
            self.features = joblib.load(f)
            self.features = self.features['lmel']  # FIXME: parameterise this!
        
        # the followig check should also be done for the annotations (might be more)
        self.features = {track_id: features for track_id, features 
                         in self.features.items() if track_id in self.track_ids}
        
        self.transform = transform


    def __len__(self):
        return len(self.track_ids)
    
    
    def __getitem__(self, idx):
        
        track_id = self.track_ids[idx]
        features = self.features[track_id]
        annotation = self.annotations[idx]

        if self.transform:
            return self.transform(features, annotation)

        return features, annotation
    

class MultiSourceStaticDataset(StaticDataset):
    """
    MuStatic dataset for summative music emotion recognition in the Valence-Arousal space.
    """

    def __init__(self, features_fname, annotations_fname, source_names, track_id="track_id", transform=None):
        """
        Args:
            features_fname (str): path to the audio features per source in a compressed file.
            annotations_fname (str): path to the csv file with the summative annotations.
            transform (callable, optional): optional transform to be applied on a sample.

        TODO:
            - also check whether the annotation track_ids are more than the song ids.
            - consider the possibility of applying the rescaler here;
        """
        assert all([s_name for s_name in source_names if s_name in SOURCE_NAMES]), \
            f"Uknown source names: choose one or more in {SOURCE_NAMES}!"

        super(MultiSourceStaticDataset, self).__init__(
            features_fname, annotations_fname, track_id, transform)
        
        # the followig check should also be done for the annotations (might be more)
        self.features = {
            track_id: np.stack([features[source_name] for source_name in source_names]) 
            for track_id, features in self.features.items()}
        

def get_partition_data_loaders(dataset, batch_sizes=[None, None, None], split_pct=[.8*.8, .8*.2, .2], num_workers=0, pin_memory=False):
    """
    Returns the data loaders for the training, validation and test sets from
    a random partitioning of the given dataset according to the split percentages.

    Args:
        dataset: a torch.dataset;
        batch_size: number of samples per batch;
        split_pct: how to split the dataset for training, validation and test sets.
        num_workers (int): ...
        pin_memory (bool): ...

    Returns:
        DataLoader: training data loader;
        DataLoader: validation data loader;
        DataLoader: testset data loader;
    """

#     batch_size = len(dataset) if batch_size == None else batch_size
    # compute the size of each dataset split
    tr_samples = len(dataset) * split_pct[0]
    va_samples = len(dataset) * split_pct[1]
    te_samples = len(dataset) * split_pct[2]
    te_samples += (tr_samples % 1) + (va_samples % 1)
    
    batch_sizes = [int(loader) if b_size is None else b_size for b_size, loader \
                   in zip(batch_sizes, [tr_samples, va_samples, te_samples])]

    # creating a partitioning of the original dataset
    train_set, valid_set, test_set = random_split(
        dataset, [int(tr_samples), int(va_samples), int(te_samples)]
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_sizes[0], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_sizes[1], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_sizes[2], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader


def create_data_loaders(devset_dataset, train_ids, valid_ids, batch_sizes=[None, None],
                        shuffle=True, num_workers=0, pin_memory=False):
    """
    Creates training and validation datasets from the parent development set (useful for
    cross validation, as we do not need to redefine new datasets for each fold).
    
    TODO: Extend with the test set...
    """

    batch_sizes = [len(ids) if b_size is None else b_size for b_size, ids \
                   in zip(batch_sizes, [train_ids, valid_ids])]

    tr_loader = DataLoader(Subset(devset_dataset, train_ids),
                           batch_size=batch_sizes[0], shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)
    va_loader = DataLoader(Subset(devset_dataset, valid_ids),
                           batch_size=batch_sizes[1], shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)

    return tr_loader, va_loader


def nested_cv_splits(dataset_ids, outer_splits=5, inner_splits=5, shuffle=True, random_state=None):
    """
    Split the dataset (from the ids) for nested cross validation.
    
    Args:
        dataset_ids (list): the ids of the data samples in the dataset;
        outer_splits (int): number of outer nested CV splits;
        inner_splits (int): number of inner nested CV splits;
        shuffle (bool): whether to shuffle data before the splits;
        random_state (int): seed for experimental reproducibility.
        
    Todo:
        - Make the printing optional.
    
    Returns  a dict with the partitioning.
    """
    
    inner_cv = KFold(n_splits=inner_splits, shuffle=shuffle, random_state=random_state)
    outer_cv = KFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state)

    nestedcv_fold_dict = {}
    for outer_fold, (devset_ids, testset_ids) in enumerate(outer_cv.split(dataset_ids)):

        print("Fold {} | Devset samples: {}: Testset samples: {}"
              .format(outer_fold, len(devset_ids), len(testset_ids)))

        nestedcv_fold_dict[outer_fold] = {'test_ids': testset_ids}
        for inner_fold, (training_ids, validation_ids) in enumerate(inner_cv.split(devset_ids)):

            print("\tInner fold {} | Training samples: {}: Validation samples: {}"
              .format(inner_fold, len(training_ids), len(validation_ids)))
            nestedcv_fold_dict[outer_fold][inner_fold] = {
                'training_ids': devset_ids[training_ids],
                'validation_ids': devset_ids[validation_ids]
            }
    
    return nestedcv_fold_dict


def load_nested_cv_fold(nested_cv_ids_path, sel_folds=None):
    """
    Load the dictionary with the ids of each nested cv split.
    Only the selected outer folds will be returned (all otherwise).
    
    Args:
        nested_cv_splits (str): path to the file with the nested cv splits;
        sel_folds (list): list of outer folds to load as integer numbers.
    """
    
    with open(nested_cv_ids_path, "rb") as f:
        ncv_dict = joblib.load(f)

    # select all the outer folds if none are specified and sanity checks
    sel_folds = list(ncv_dict.keys()) if sel_folds is None else sel_folds
    assert all(fold in ncv_dict.keys() for fold in sel_folds)

    ncv_dict_sel = {outer_fold : outer_fold_data for outer_fold, outer_fold_data 
                    in ncv_dict.items() if outer_fold in sel_folds}
    
    return ncv_dict_sel