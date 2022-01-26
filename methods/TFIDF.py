import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher

from utils import readFasta, checkFasta

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)


class TFIDF(object):
    def __BLOSUM62__(self, fastas, **kw):
        if not checkFasta.checkFasta(fastas):
            print('Error: for "BLOSUM62" encoding, the input fasta sequences should be with equal length. \n\n')
            return 0
        encodings = []
        h = FeatureHasher(n_features=100)
        header = ['#']
        for i in range(1, len(fastas[0][1]) * 20 + 1):
            header.append('blosum62.F' + str(i))
        encodings.append(header)
        for i in fastas:
            name, sequence = i[0], i[1]
            code = [name]
            countsNEw = dict()
            namesNEw = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                        'V']
            for nameNEw in namesNEw:
                countsNEw[nameNEw] = sequence.count(nameNEw)
            f = h.transform([countsNEw])
            # print(f.toarray().tolist()[0])
            encodings.append(f.toarray().tolist()[0])
        return encodings

    def min_max(self, df: pd.DataFrame):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

    def do(self, input_files: list, output_files: list):
        kw = {'path': input_files[0], }
        fastas1 = readFasta.readFasta(input_files[0])
        result = self.__BLOSUM62__(fastas1, **kw)
        data1 = np.matrix(result[1:])[:, 1:]
        data_ = pd.DataFrame(data=data1)
        data_ = self.min_max(data_)
        data_.to_csv(output_files[0])
