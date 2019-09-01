import py_entitymatching as em
import os
import pandas as pd

datasets_dir = em.get_install_path() + os.sep + 'datasets'
path = datasets_dir + os.sep + 'dblp_demo.csv'

A = em.read_csv_metadata(path, key='id')
B = em.read_csv_metadata(path, key='id')
print(A.head())
em.data_explore_pandastable(B)


