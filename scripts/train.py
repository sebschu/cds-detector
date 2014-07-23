#!/usr/bin/python

'''
    Trains different classifiers on the Emobase 2010 dataset
'''


# Add path to our module(s).
import sys
import os
import pickle
import argparse

sys.path.append("../src/python")

import motherese

# Paths.
FEATS_DIR = "../feats"
MODEL_DIR = "../models"

import numpy as np

from sklearn import svm
from sklearn import cluster
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import tree
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import metrics


def main(config_name, prefix,unbalancetrain):
    
    print "Creating models with config: \"{}\"".format(config_name)
    
    #create models dir
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    #### Loading data    
    data = motherese.load_data(config_name, True, (not unbalancetrain), data_dir=FEATS_DIR)
    entire_data = motherese.load_data(config_name, True, False, data_dir=FEATS_DIR)
    ### save scaler
    
    pickle.dump(data["scaler"], open(os.path.join(MODEL_DIR, prefix + "scaler.p"), 'wb'))
    
    
    
    #### Support vector machines and leafy classifiers
    
    
    classifier_list = [
        ("libsvm", lambda: svm.SVC(kernel='linear', class_weight='auto', C=0.001)),
        ("logreg", lambda: svm.LinearSVC(C=0.01, penalty="l1", dual=False, class_weight='auto')),
        ("rbf_svm", lambda: svm.SVC(kernel='rbf')),
        ("rbf_nu_svm", lambda: svm.NuSVC(kernel='rbf')),
        ("random_forest", lambda: ensemble.RandomForestClassifier(criterion="entropy", n_estimators=50, min_samples_split=3, max_depth=6)),
        ("decision_tree", lambda: tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=3, max_depth=6)),
        ("naive_bayes", lambda: naive_bayes.GaussianNB())
    ]

    for name, clf_gen in classifier_list:
        try:
            clf = clf_gen()
            clf.fit(data['X_train'], data['y_train'])
        except:
            print 'Skipping ' + name
            continue

        # CHANGE NAME 
        pickle.dump(clf, open(os.path.join(MODEL_DIR, prefix + name + ".p"), 'wb'))
        predictions = clf.predict(data['X_train'])
        predictions_test = clf.predict(entire_data['X_test'])
        print name, metrics.roc_auc_score(data['y_train'], predictions), metrics.roc_auc_score(data['y_test'], predictions_test),\
          metrics.accuracy_score(data['y_train'], predictions), metrics.accuracy_score(data['y_test'], predictions_test), \
          metrics.precision_score(data['y_train'], predictions), metrics.precision_score(data['y_test'], predictions_test), \
          metrics.recall_score(data['y_train'], predictions), metrics.recall_score(data['y_test'], predictions_test), \
          metrics.f1_score(data['y_train'], predictions), metrics.f1_score(data['y_test'], predictions_test), \
          sum(data['y_train']==0), sum(data['y_train']==1), \
          sum(data['y_test']==0), sum(data['y_test']==1)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train motherese classifiers.")
    parser.add_argument("config_name", help="Configuration to use (e.g. hw4, default: emobase2010-nomfcc)", 
                        default="emobase2010-nomfcc", nargs="?")
    parser.add_argument("prefix",help="Prefix for model name",default="", nargs="?")
    parser.add_argument("--unbalancetrain", action="store_true",default=False)
    args = parser.parse_args()

    main(args.config_name, args.prefix, args.unbalancetrain)
