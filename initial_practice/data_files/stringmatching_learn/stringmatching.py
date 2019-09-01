import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd
import os, sys

print('python version: ' + sys.version)
print('py_stringsimjoin version: ' + ssj.__version__)
print('py_stringmatching version: ' + sm.__version__)
print('pandas version: ' + pd.__version__)

table_A_path = os.sep.join([ssj.get_install_path(), 'datasets', 'data', 'person_table_A.csv'])
table_B_path = os.sep.join([ssj.get_install_path(), 'datasets', 'data', 'person_table_B.csv'])


# Load csv files as dataframes.
A = pd.read_csv(table_A_path)
B = pd.read_csv(table_B_path)
print('Number of records in A: ' + str(len(A)))
print('Number of records in B: ' + str(len(B)))
ssj.profile_table_for_join(A)

B['new_key_attr'] = range(0, len(B))
ws = sm.WhitespaceTokenizer(return_set=True)
ws.tokenize('William Bridge')
output_pairs = ssj.edit_distance_join(A, B, 'A.id', 'B.id', 'A.name', 'B.name', 5,
                                l_out_attrs=['A.name'], r_out_attrs=['B.name'])
print(len(output_pairs))
print(output_pairs)

