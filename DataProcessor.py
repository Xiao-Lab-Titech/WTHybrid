"""
IN:../THINCData/casex_y.txt
OUT:MTBVDdata.txt

COMMENT:
Processing raw data into dataset.
The dataset include training and test data.

<INPUT FILE ITEMS>
input file has 14 items
this code use <>
<u_k-2> <u_k-1> <u_k> <u_k+1> <u_k+2> 
u_k-1-u_k-2/dx u_k-u_k-1/dx u_k+1-u_k/dx u_k+2-u_k+1/dx
<sign(xi)> sign_vof_max sign_vof_min
cell_width <scheme_indicator(y)>

"""

import torch
import numpy as np
import os
from collections import namedtuple
import random

#--------------#
#   FUNCTION   #
#--------------#

# Set a random seed
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# Load raw data
def load_data(file_par_path):
    file_list = os.listdir(file_par_path)
    lines = []
    for file_path in file_list:
        with open(os.path.join(file_par_path, file_path), "r") as f:
            lines += f.readlines()
    return lines

# Return a 2d-list with duplicate data removed
def filter_dump_data(lines):
    data_set = set(lines)
    data_list = list(map(lambda x: x.replace('\n', '').split('\t'), data_set))
    for idx, l in enumerate(data_list):
        data_list[idx] = list(map(float, l))
    return data_list

#--------------#
#  PARAMETER   #
#--------------#

#input_files = "./PreProcessedData/" # raw data
input_files = "./THINCData/" # raw data
#utput_file = "./PostProcessedData/W3TBVDdataset.dat" # training data
output_file = "./PostProcessedData/MTBVDdataset.dat" # training data
N_dataset_case0 = 5000 # # of WENO data
N_dataset_case1 = 5000 # # of THINC data

#--------------#
#  MAIN CODE   #
#--------------#

# Setting GPU by Pytorch
print("Loading device...",end="")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch.device("mps") # for mac
print("Recognize")
#print("# of GPU: {}".format(torch.cuda.device_count()))
#print("Num_0 of GPU Name: {}".format(torch.cuda.get_device_name(torch.device("cuda:0"))))

print("Setting seed...",end="")
setup_seed(1234)
print("OK")

print("Loading raw data...",end="")
lines = load_data(input_files)
print("OK")

data_list = filter_dump_data(lines)
Data = namedtuple("Data", ["data", "sign", "dh", "target"])
data = Data(np.array(data_list)[:, :-5], 
            np.array(data_list)[:,-5:-2],
            np.array(data_list)[:,-2],
            np.array(data_list)[:,-1])
X = data.data
sign = data.sign
h = data.dh.reshape(-1, 1)
y = data.target.reshape(-1,1)
#print(X.shape, sign.shape, h.shape, y.shape, X[0,:], y)

print("{} raw data loaded".format(len(y)))

number_1 = int(np.sum(y))
number_0 = len(y) - number_1
print(f"# of case0: { number_0}, # of case1: { number_1}")

print("\nMaking dataset")
x1 = X[:,0:5]
#print(sign[0,:])

print("1) Discarding all-zero stencil data...",end="")
idx_zero = []
idx_ones = []
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        if abs(x1[i][j]) < 1.e-16:
            x1[i][j] = 0.0
for i in range(x1.shape[0]):
    M = np.max(np.fabs(x1[i]))
    if abs(M) < 1.e-16:
        idx_zero.append(i)
Idx = [i for i in range(x1.shape[0])]
idx1 = list(set(Idx) - set(idx_zero)) # Discard all 0 stencil value
print("OK")

x2 = x1[idx1,:]
sign2 = sign[idx1,0]
y2 = y[idx1,:]

print("\nSize of dataset: {}".format(len(y2)))
number_1 = int(np.sum(y2))
number_0 = len(y2) - number_1
print(f"# of case0: { number_0}, # of case1: { number_1}")

print("2) Devide raw data by candidate function...",end="")
for i in range(x2.shape[0]):
    if abs(y2[i][0] - 1.) < 1.e-16:
        idx_ones.append(i) 
Idx2 = [i for i in range(x2.shape[0])]
idx2 = list(set(Idx2) - set(idx_ones)) # Discard label 1 data
print("OK")
"""
print("3) Normalization...",end="")
# Normalization
M = np.max(x2,axis=1)
m = np.min(x2,axis=1)

for i in range(x2.shape[0]):
    if (M[i] - m[i] < 1.e-16):
        x2[i] = [0.0,0.0,0.0,0.0,0.0]
    else:
        x2[i] = (x2[i] - m[i])/(M[i] - m[i])
print("OK")
"""
# Ramdomize devided data and unite as dataset without duplicating.
random.shuffle(idx_ones)
random.shuffle(idx2)
idx = list(set.union(set(idx_ones[0:5000]), set(idx2[0:5000])))

x_print = np.hstack([x2[idx,:], sign2[idx].reshape(-1,1), y2[idx].reshape(-1,1)])

# Print detail of dataset
print("Size of dataset: {}".format(len(x_print)))
number_1 = int(np.sum(y2[idx]))
number_0 = len(y2[idx]) - number_1
print(f"# of case0: { number_0}, # of case1: { number_1}")


# Save dataset
Data_save = x_print
train_file_name = output_file
np.savetxt(train_file_name, Data_save)
print("Save dataset")
