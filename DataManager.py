import pandas
import os.path

class DataManager():
    def __init__(self):
        self.read_data()

    def read_data(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "dataset_noclass.csv")
        cols = ['x', 'y', 'z']
        self.data = pandas.read_csv(path, index_col=False, names=cols)
        self.data = self.data.iloc[1:]  # deletes header row from file
        self.data = self.data.astype(float)
        self.data = self.data.values
# 
#
# data = DataManager()
# print(data.data)
