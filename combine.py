import os
import pandas as pd


def combine(files: list, output_file_name: str):
    print('combining...')
    print(files)
    csv = []
    for file in files:
        if len(file) == 2:
            csv.append(pd.read_csv(file[0]))
        else:
            csv.append(pd.read_csv(file))

    concatenated = pd.concat(csv, axis=1)
    concatenated.drop('Unnamed: 0', inplace=True, axis=1)
    concatenated.drop(1, 0)

    concatenated.to_csv(output_file_name)
    print('combined')
