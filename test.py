import os
import pandas as pd
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

data1 = pd.read_csv('./data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt', sep='\t')
data2 = pd.read_csv('./data_bcr_clinical_data_patient.txt', sep='\t')
data1 = data1.set_index('Hugo_Symbol', drop = True)

data1 = data1.T
data1.rename(index = lambda x : str(x).replace('-01', ''), inplace = True)
data1 = data1.iloc[1:]

data2 = data2.iloc[4:]
data2 = data2[['Patient Identifier', 'Overall Survival Status', 'Overall Survival (Months)']]
data2 = data2.set_index('Patient Identifier', drop = True)

data = data2.merge(data1, left_on = 'Patient Identifier', right_index = True)

missing_data = data.isnull().sum() / len(data) > 0.8
missing_data = missing_data.to_frame()

missing_data_index = missing_data.loc[missing_data[0] == True, :].index
data = data.drop(missing_data_index, axis = 1)

dup_cols = data.columns[data.columns.duplicated()]
dup_indices = [ [i, col] for i, col in enumerate(data.columns) if col in dup_cols]
dup_indices.sort(key=lambda x:x[1])
dup_indices = list(list(zip(*dup_indices))[0])
indexes = []

for i in range(0, len(dup_indices), 2):
    pair = data.iloc[:, dup_indices[i:i+2]]
    variances = pair.var()   
    if variances.iloc[0] < variances.iloc[1]:
        indexes.append(dup_indices[i])        
    else:
        indexes.append(dup_indices[i+1])
data = data.iloc[:, [i for i in range(data.shape[1]) if i not in indexes]]

data['Overall Survival Status'] = data['Overall Survival Status'].replace({'1:DECEASED': 1, '0:LIVING': 0})
data = data.rename(columns={'Overall Survival (Months)': 't', 'Overall Survival Status': 'e'})
data['t'] = data['t'].replace({'[Not Available]': 0})

corrIndex = pd.read_csv('./corrIndex.txt', sep = '\t', header = None)
corrIndex.drop(corrIndex.columns[7091], axis = 1)
corrIndex = corrIndex.values.flatten()
corrIndex = np.append(corrIndex, 'e')
corrIndex = np.append(corrIndex, 't')

from sklearn.model_selection import train_test_split
train, test = train_test_split(data[corrIndex], test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25,random_state=42)

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()

t_train = train['t'].to_numpy(dtype = 'float')
e_train = train['e'].to_numpy(dtype = 'int')
X_train = train.drop(['t', 'e', 'Patient Identifier'], axis = 1)

t_val = val['t'].to_numpy(dtype = 'float')
e_val = val['e'].to_numpy(dtype = 'int')
X_val = val.drop(['t', 'e', 'Patient Identifier'], axis = 1)

t_test = test['t'].to_numpy(dtype = 'float')
e_test = test['e'].to_numpy(dtype = 'int')
X_test = test.drop(['t', 'e', 'Patient Identifier'], axis = 1)

X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)
X_val = X_val.to_numpy(dtype=float)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

import sys
import os

sys.path.append('/home/spark/Definition-of-Pathological-Image-Analysis-for-Cancer-Cla/SurvivalNet/survivalnet')
sys.path.append('/home/spark/Definition-of-Pathological-Image-Analysis-for-Cancer-Cla/SurvivalNet')

from survivalnet import train

from survivalnet.optimization import SurvivalAnalysis

train_set = {}
test_set = {}
val_set = {}

sa = SurvivalAnalysis()

train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(
    X_train,t_train, e_train);
test_set['X'], test_set['T'], test_set['O'], test_set['A'] = sa.calc_at_risk(
    X_test,t_test,e_test);
val_set['X'], val_set['T'], val_set['O'], val_set['A'] = sa.calc_at_risk(
    X_val,t_val,e_val);

epochs = 10000
finetune_config = {'ft_lr':0.0001, 'ft_epochs':epochs}
n_layers = 1
n_hidden = 100
do_rate = 0.5
lambda1 = 0
lambda2 = 0
nonlin = np.tanh
opt = 'GDLS'
pretrain_config = None

train_costs, train_cindices, test_costs, test_cindices, train_risk, test_risk, model, max_iter = train(X_train, train_set, val_set, pretrain_config,finetune_config, n_layers, n_hidden, dropout_rate = do_rate,lambda1 = lambda1, lambda2 = lambda2, non_lin = nonlin,optim = opt, verbose = True, earlystp = False)

print(train_costs)