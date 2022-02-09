import os
import pandas as pd

from tqdm import tqdm


class DataGenerator(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        columns
    ):
        self.columns = columns

    # Initial Conditions
    ###########################################################################
    def initial_cond(
        self
    ):
        pass

    # Training/Validation Data
    ###########################################################################
    def training_data(
        self,
        ic_train=None
    ):
        pass

    # Testing Data
    ###########################################################################
    def testing_data(
        self,
        ic_test=None,
        header=None
    ):
        if isinstance(ic_test, str):
            files = [file for file in os.scandir(ic_test)]
            data = []
            for file in tqdm(files):
                if file.is_file() and os.path.splitext(file)[1] == '.csv':
                    df = pd.read_csv(file, sep=',', header=header).to_numpy()
                    data.append([
                        self.read_dataframe(df, self.columns[i]) \
                            for i in ['x', 'y']
                    ])
        else:
            NotImplementedError
        return data

    def read_dataframe(self, df, col):
        if any(isinstance(i, (list,tuple)) for i in col):
            return [df[:,col[i][0]:col[i][1]] for i in range(len(col))]
        else:
            return df[:,col[0]:col[1]]
