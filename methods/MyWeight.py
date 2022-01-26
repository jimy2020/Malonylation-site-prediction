import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

from utils import readFasta

pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)


class Weight(object):
    def __weight__(self, fastas, fastas1):
        positionValue = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                         'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': 20}
        encodings = []
        header = ['#']
        for i in range(1, len(fastas[0][1]) * 20 + 1):
            header.append('positionValue.F' + str(i))
        encodings.append(header)
        ####################################
        encodings1 = []
        header1 = ['#']
        for i in range(1, len(fastas1[0][1]) * 20 + 1):
            header1.append('positionValue.F' + str(i))
        encodings1.append(header)
        ###################################
        resultposclass1 = np.zeros((21, 25))
        resultposclass2 = np.zeros((21, 25))
        numersample = 0
        for i in fastas:
            classNum = 2
            name, sequence = i[0], i[1]
            code = [name]
            j = 0
            if numersample < len(fastas) / 2:
                for aa in sequence:
                    indexchar = positionValue[aa]
                    resultposclass1[indexchar][j] = resultposclass1[indexchar][j] + 1
                    j = j + 1
            else:
                for aa in sequence:
                    indexchar = positionValue[aa]
                    resultposclass2[indexchar][j] = resultposclass2[indexchar][j] + 1
                    j = j + 1
            numersample = numersample + 1
        # print(resultposclass1)
        # print(resultposclass2)
        resultCrfclass1 = np.zeros((21, 25))
        resultCrfclass2 = np.zeros((21, 25))
        for i in range(0, 21):
            for j in range(0, 25):
                resultCrfclass1[i][j] = resultposclass1[i][j] / (resultposclass2[i][j] + 0.00001)
                resultCrfclass2[i][j] = resultposclass2[i][j] / (resultposclass1[i][j] + 0.00001)

        numersample = 0
        for ii in fastas:
            resultFinal = np.zeros(25)
            name, sequence = ii[0], ii[1]
            jj = 0
            for aa in sequence:
                indexchar = positionValue[aa]
                if numersample < len(fastas) / 2:
                    resultFinal[jj] = resultCrfclass1[indexchar][jj]
                else:
                    resultFinal[jj] = resultCrfclass2[indexchar][jj]
                jj = jj + 1
            numersample = numersample + 1
            encodings.append(resultFinal)

        ########################################
        numersample = 0
        for ii in fastas1:
            resultFinal = np.zeros(25)
            name, sequence = ii[0], ii[1]
            jj = 0
            for aa in sequence:
                indexchar = positionValue[aa]
                if numersample < len(fastas) / 2:
                    resultFinal[jj] = resultCrfclass1[indexchar][jj]
                else:
                    resultFinal[jj] = resultCrfclass2[indexchar][jj]
                jj = jj + 1
            numersample = numersample + 1
            encodings1.append(resultFinal)
        return encodings, encodings1

    def min_max(self, df: pd.DataFrame):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

    def do(self, input_files: list, output_files: list):
        kw1 = {'path': input_files[0], }
        fastas1 = readFasta.readFasta(input_files[0])
        kw2 = {'path': input_files[1], }
        fastas2 = readFasta.readFasta(input_files[1])

        result, result2 = self.__weight__(fastas1, fastas2)
        data1 = np.matrix(result[1:])[:, 1:]
        data_ = pd.DataFrame(data=data1)
        data_ = self.min_max(data_)
        data_.to_csv(output_files[0])

        data2 = np.matrix(result2[1:])[:, 1:]
        data_2 = pd.DataFrame(data=data2)
        data_2 = self.min_max(data_2)
        data_2.to_csv(output_files[1])
