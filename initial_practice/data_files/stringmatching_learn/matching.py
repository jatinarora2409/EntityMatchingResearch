import py_entitymatching as em
import os
import pandas as pd


# Set the seed value
seed = 0
datasets_dir = em.get_install_path() + os.sep + 'datasets'

path_A = datasets_dir + os.sep + 'dblp_demo.csv'
path_B = datasets_dir + os.sep + 'acm_demo.csv'
path_labeled_data = datasets_dir + os.sep + 'labeled_data_demo.csv'

A = em.read_csv_metadata(path_A, key='id')
B = em.read_csv_metadata(path_B, key='id')
# Load the pre-labeled data
S = em.read_csv_metadata(path_labeled_data,
                         key='_id',
                         ltable=A, rtable=B,
                         fk_ltable='ltable_id', fk_rtable='rtable_id')

IJ = em.split_train_test(S, train_proportion=0.5, random_state=0)
I = IJ['train']
J = IJ['test']

dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')

F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)

H = em.extract_feature_vecs(I,
                            feature_table=F,
                            attrs_after='label',
                            show_progress=False)
print(any(pd.notnull(H)))
# Impute feature vectors with the mean of the column values.
H = em.impute_table(H,
                exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                strategy='mean')


