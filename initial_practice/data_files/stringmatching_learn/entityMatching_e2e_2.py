import sys

sys.path.append('/home/jatinarora2409/EntityMatchingResearch/initial_practice/py_entitymatching/py_entitymatching')

import py_entitymatching as em
import pandas as pd
import os


# Reading Database
path_A = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/fodors.csv'
path_B = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/zagats.csv'

A = em.read_csv_metadata(path_A,key='id')
B = em.read_csv_metadata(path_B,key='id')
em.down_sample(A,B,200,1,show_progress=False)

print('Number of tuples in A' + str(len(A)))
print('Number of tuples in B' + str(len(B)))
print('Number of tuples in A X B: ' + str(len(A)*len(B)))

ob = em.OverlapBlocker()
C = ob.block_tables(A,B,'name','name',overlap_size=1,l_output_attrs=['name','addr','city','phone'],r_output_attrs=['name','addr','city','phone'])
print(len(C))


S = em.sample_table(C,450)

path_G = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/lbl_restnt_wf1.csv'
G = em.read_csv_metadata(path_G,
                         key='_id',
                         ltable=A, rtable=B,
                         fk_ltable='ltable_id', fk_rtable='rtable_id')

IJ = em.split_train_test(G,train_proportion=0.7,random_state=0)
I = IJ['train']
J=IJ['test']

dt = em.DTMatcher(name='DecisionTree',random_state=0)
svm=em.DTMatcher(name='SVM',random_state=0)
rf = em.RFMatcher(name='RF',random_state=0)
lg=em.LogRegMatcher(name='LogReg',random_state=0)
ln=em.LinRegMatcher(name='LinReg')
nb=em.NBMatcher(name='NaiveBayes')


feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
print(feature_table)

H = em.extract_feature_vecs(I,
                            feature_table=feature_table,
                            attrs_after='gold',
                            show_progress=False)


result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H,
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
        k=5,
        target_attr='gold', metric_to_select_matcher='precision', random_state=0)
print(result['cv_stats'])

result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H,
        exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
        k=5,
        target_attr='gold', metric_to_select_matcher='recall', random_state=0)
print(result['cv_stats'])



