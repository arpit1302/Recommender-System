import time

import pandas as pd
import numpy as np
import pickle

USER_ID = 0
MOVIE_ID = 1
RATINGS = 2
class PreProcessData:
    """
    Helper class to structure dataset. Save's final matrix into a .npy file
    """

    def __init__(self, filename=None):
        """
        Class initialized with with data from given dataset
        """
        self.filename = filename
        self.user_count = 943
        self.movie_count = 1682

    def read_data(self, filename):
        """
        Returns a pandas dataframe of the dataset with columns labelled as 0,1,2.
        """
        df = pd.read_csv(filename, '\t', header=None)
        return df

    def PreProcess(self, limit_users=None):
        """
        Initializes an output matrix and fills it with ratings from the dataset.

        Input:
        number of entries in the dataset to be considered

        Output:
        Dataframe and a numpy array saved as 'data_df.csv' and 'data_np.npy' respectively.
        """
        clean_start_time = time.time()
        print("Formatting dataset")
        df = self.read_data(self.filename)
        data = np.zeros([self.user_count, self.movie_count])
        for i in range(df.shape[0]):
            data[df.iloc[i][USER_ID]-1][df.iloc[i][MOVIE_ID]-1] = df.iloc[i][RATINGS]
        np.save('data.npy', data)
        print("Formatted dataset.")
        print("Time to format dataset: " + str(time.time() - clean_start_time))


if __name__=="__main__":
    cleaner = CleanData('ml-100k/u.data')
    cleaner.process()
