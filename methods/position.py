import numpy as np
import pandas as pd


class Position(object):
    def RetriveFeatureFromASequence(self, seq):
        seq = seq.rstrip('\n').rstrip(' ')
        assert (len(seq) >= 2)
        Feature = []
        for index, item in enumerate(seq):
            Feature.append(float(index + 1) / float(len(seq)))
        return Feature

    def load_fasta_and_compute(self, input_file):
        encodings = []
        fin = open(input_file, "r")

        while True:
            line_Pid = fin.readline()
            line_Pseq = fin.readline()
            if not line_Pseq:
                break
            # fout.write(line_Pid)
            # fout.write(line_Pseq)
            Feature = self.RetriveFeatureFromASequence(line_Pseq)
            # fout.write(",".join(map(str,Feature)) + "\n")
            encodings.append(Feature)
        fin.close()
        return encodings

    def do(self, input_files: list, output_files: list):
        result = self.load_fasta_and_compute(input_files[0])
        data1 = np.matrix(result[:])
        data_ = pd.DataFrame(data=data1)
        data_.to_csv(output_files[0])
