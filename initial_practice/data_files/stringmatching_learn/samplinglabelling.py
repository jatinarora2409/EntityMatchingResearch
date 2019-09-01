# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd


# Get the datasets directory
datasets_dir = em.get_install_path() + os.sep + 'datasets'

path_A = datasets_dir + os.sep + 'DBLP.csv'
path_B = datasets_dir + os.sep + 'ACM.csv'
path_C = datasets_dir + os.sep + 'tableC.csv'


A = em.read_csv_metadata(path_A, key='id')
B = em.read_csv_metadata(path_B, key='id')
C = em.read_csv_metadata(path_C, key='_id',
                         fk_ltable='ltable_id', fk_rtable='rtable_id',
                         ltable=A, rtable=B)
S = em.sample_table(C, 450)

path_labeled_data = datasets_dir + os.sep + 'labeled_data_demo.csv'
G = em.read_csv_metadata(path_labeled_data, key='_id',
                         fk_ltable='ltable_id', fk_rtable='rtable_id',
                         ltable=A, rtable=B)

print(G.head())