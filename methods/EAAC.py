import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import preprocessing

from utils import readFasta, checkFasta

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)


class EAAC(object):
    def __EAAC__(self, fastas, window=5, **kw):
        if not checkFasta.checkFasta(fastas):
            print('Error: for "EAAC" encoding, the input fasta sequences should be with equal length. \n\n')
            return 0

        if window < 1:
            print('Error: the sliding window should be greater than zero' + '\n\n')
            return 0

        if checkFasta.minSequenceLength(fastas) < window:
            print('Error: all the sequence length should be larger than the sliding window :' + str(window) + '\n\n')
            return 0

        AA = kw['order'] if kw['order'] is not None else 'ACDEFGHIKLMNPQRSTVWY'
        # AA = 'ARNDCQEGHILKMFPSTWYV'
        encodings = []
        header = ['#']
        for w in range(1, len(fastas[0][1]) - window + 2):
            for aa in AA:
                header.append('SW.' + str(w) + '.' + aa)
        encodings.append(header)

        for i in fastas:
            name, sequence = i[0], i[1]
            code = [name]
            for j in range(len(sequence)):
                if j < len(sequence) and j + window <= len(sequence):
                    count = Counter(re.sub('-', '', sequence[j:j + window]))
                    for key in count:
                        count[key] = count[key] / len(re.sub('-', '', sequence[j:j + window]))
                    for aa in AA:
                        code.append(count[aa])
            encodings.append(code)
        return encodings

    def min_max(self, df: pd.DataFrame):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

    def do(self, input_files: list, output_files: list):
        kw = {'path': input_files[0], 'order': 'ACDEFGHIKLMNPQRSTVWY'}
        fastas = readFasta.readFasta(input_files[0])
        sw = 5
        result = self.__EAAC__(fastas, sw, **kw)
        data1 = np.matrix(result[1:])[:, 1:]
        data_ = pd.DataFrame(data=data1)
        data_ = self.min_max(data_)
        data_.to_csv(output_files[0])
