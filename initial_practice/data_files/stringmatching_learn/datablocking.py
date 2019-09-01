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

ob = em.OverlapBlocker()

# Apply blocking to a tuple pair from the input tables on zipcode and get blocking status
status = ob.block_tuples(A.loc[0], B.loc[0],'address', 'address', overlap_size=4)

# Print the blocking status
print(status)

