# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd

# Get the datasets directory
datasets_dir = em.get_install_path() + os.sep + 'datasets'

# Get the paths of the input tables
path_A = datasets_dir + os.sep + 'person_table_A.csv'
path_B = datasets_dir + os.sep + 'person_table_B.csv'

A = em.read_csv_metadata(path_A, key='ID')
B = em.read_csv_metadata(path_B, key='ID')

# print(A.head())
feature_table = em.get_features_for_blocking(A, B)
print(feature_table.head())

sim = em.get_sim_funs_for_matching()
tok = em.get_tokenizers_for_matching()
feature_string = """jaccard(wspace((ltuple['name'] + ' ' + ltuple['address']).lower()), 
                            wspace((rtuple['name'] + ' ' + rtuple['address']).lower()))"""
feature = em.get_feature_fn(feature_string, sim, tok)

# Add feature to F
em.add_feature(feature_table, 'jac_ws_name_address', feature)