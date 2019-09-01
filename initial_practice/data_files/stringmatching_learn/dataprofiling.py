import py_entitymatching as em
import os
import pandas as pd
import pandas_profiling

datasets_dir = em.get_install_path() + os.sep + 'datasets'
path_A = datasets_dir + os.sep + 'dblp_demo.csv'
A = em.read_csv_metadata(path_A, key='id')
print(A.head())
pfr = pandas_profiling.ProfileReport(A)
pfr.to_file("./example.html")



