"""
Motherese utility functions.
"""

import csv
import pickle
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils import safe_asarray

import numpy as np
import scipy as sp


MODEL_DIR = "../models"


def load_data(config, normalize, balance, data_dir=None):
    """Load motherese dataset (train and test).

    :config: Name of configuration used to extract features.
    :normalize: Whether to normalize data (performs standard scaling).
    :balance: Whether to balance the data or not (randomly sample an equal amount from both of them).
    :returns: a dictionary with four keys:
        ['X_train', 'y_train', 'X_test', y_test']

    """
    with open(os.path.join(data_dir, "train-{}.p".format(config)), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(data_dir, "test-{}.p".format(config)), 'rb') as f:
        test = pickle.load(f)

    scaler = None
    if normalize:
        scaler = preprocessing.StandardScaler().fit(train["feats"])
        X_train = scaler.transform(train["feats"])
        X_test = scaler.transform(test["feats"])
    else:
        X_train = train["feats"]
        X_test = test["feats"]

    
    if balance:
        pos_examples = sum(train['labels'] == 1)
        neg_examples = sum(train['labels'] == 0)
        
        num_samples = min(pos_examples, neg_examples)
        X_neg_all = np.array([e for i, e in enumerate(X_train) if train['labels'][i] == 0])
        X_pos_all = np.array([e for i, e in enumerate(X_train) if train['labels'][i] == 1])
        
        neg_idcs_path = os.path.join(MODEL_DIR, "neg_idcs.p")
        if not os.path.isfile(neg_idcs_path):
            neg_idcs = np.random.choice(neg_examples, num_samples, False)
            pickle.dump(neg_idcs, open(neg_idcs_path, 'wb'))
        else:
            neg_idcs = pickle.load(open(neg_idcs_path, 'rb'))

        pos_idcs_path = os.path.join(MODEL_DIR, "pos_idcs.p")
        if not os.path.isfile(pos_idcs_path):
            pos_idcs = np.random.choice(pos_examples, num_samples, False)
            pickle.dump(pos_idcs, open(pos_idcs_path, 'wb'))
        else:
            pos_idcs = pickle.load(open(pos_idcs_path, 'rb'))
        
        X_neg = X_neg_all[neg_idcs, :]
        X_pos = X_pos_all[pos_idcs, :]
        
        X_train = np.concatenate((X_neg, X_pos))
        train['labels'] = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))
        
    return {"X_train": X_train, "y_train": train["labels"],
            "X_test": X_test, "y_test": test["labels"], "scaler": scaler}

