import argparse
import random
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Make LookUp Table for Error Correction Layer')
parser.add_argument('-e', '--err', dest='err', type=str, default="0.1",
                    help='Error rate in ECL')
args = parser.parse_args()

error_rate = float(args.err)
LUT = []
zeros = []
for i in tqdm(range(0,30000000), ncols=100):
    a = 0
    for x in range(0,31):
        rand = random.uniform(0, 1)

        if rand <= error_rate:
            a = a + 1 * (2 ** x)
        else:
            a = a + 0

    LUT.append(a)
    if a == 0:
        zeros.append(0)

print(len(LUT))
print(len(zeros))

LUT = np.array((LUT))
np.savez_compressed('LUT1_{}.npz'.format(args.err), LUT=LUT)
