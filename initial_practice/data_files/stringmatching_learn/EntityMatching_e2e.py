import sys

sys.path.append('/home/jatinarora2409/EntityMatchingResearch/initial_practice/py_entitymatching/py_entitymatching')

import py_entitymatching as em
import pandas as pd
import os


# Display the versions
print('python version: ' + sys.version )
print('pandas version: ' + pd.__version__ )
print('magellan version: ' + em.__version__ )


path_A = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/fodors.csv'
path_B = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/zagats.csv'

A = em.read_csv_metadata(path_A, key='id')
B = em.read_csv_metadata(path_B, key='id')

print('Number of tuples in A: ' + str(len(A)))
print('Number of tuples in B: ' + str(len(B)))
print('Number of tuples in A X B (i.e the cartesian product): ' + str(len(A)*len(B)))


ob = em.OverlapBlocker()
C = ob.block_tables(A, B, 'name', 'name',
                    l_output_attrs=['name', 'addr', 'city', 'phone'],
                    r_output_attrs=['name', 'addr', 'city', 'phone'],
                    overlap_size=1, show_progress=False)


print('Number of tuples After blocked: ' + str(len(C)))
S = em.sample_table(C, 450)

print('Number of tuples in Sample: ' + str(len(S)))


path_G = em.get_install_path() + os.sep + 'datasets' + os.sep + 'end-to-end' + os.sep + 'restaurants/lbl_restnt_wf1.csv'
G = em.read_csv_metadata(path_G,
                         key='_id',
                         ltable=A, rtable=B,
                         fk_ltable='ltable_id', fk_rtable='rtable_id')

print('Number of tuples in Labelled: ' + str(len(G)))

feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)


# Select the attrs. to be included in the feature vector table
attrs_from_table = ['ltable_name', 'ltable_addr', 'ltable_city', 'ltable_phone',
                    'rtable_name', 'rtable_addr', 'rtable_city', 'rtable_phone']

H = em.extract_feature_vecs(G,
                            feature_table=feature_table,
                            attrs_before = attrs_from_table,
                            attrs_after='gold',
                            show_progress=False)


rf = em.RFMatcher()

attrs_to_be_excluded = []
attrs_to_be_excluded.extend(['_id', 'ltable_id', 'rtable_id', 'gold'])
attrs_to_be_excluded.extend(attrs_from_table)

rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold')

L = em.extract_feature_vecs(C, feature_table=feature_table,
                          attrs_before=attrs_from_table,
                          show_progress=False, n_jobs=-1)

attrs_to_be_excluded = []
attrs_to_be_excluded.extend(['_id', 'ltable_id', 'rtable_id'])
attrs_to_be_excluded.extend(attrs_from_table)


predictions = rf.predict(table=L, exclude_attrs=attrs_to_be_excluded,
              append=True, target_attr='predicted', inplace=False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(predictions)