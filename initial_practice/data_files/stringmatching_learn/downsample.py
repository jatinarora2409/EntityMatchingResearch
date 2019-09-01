import py_entitymatching as em
A = em.read_csv_metadata('./citeseer.csv',low_memory=False) # setting the parameter low_memory to False  to speed up loading.
B = em.read_csv_metadata('./dblp.csv', low_memory=False)
len(A), len(B)
em.set_key(A, 'id')
em.set_key(B, 'id')
em.get_key(A), em.get_key(B)
sample_A, sample_B = em.down_sample(A, B, size=1000, y_param=1, show_progress=True)





