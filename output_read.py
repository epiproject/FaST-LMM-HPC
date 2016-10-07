import pickle as pickle

file_name = "dataframe.out"
f = open(file_name, "rb")
result = pickle.load(f)

print result
