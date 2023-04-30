import pandas as pd
import numpy as np

sort_num = 100000
drop_null = 0.8

def loadData(RNA_path, patient_path) :
    data1 = pd.read_csv(RNA_path, sep='\t')
    data2 = pd.read_csv(patient_path, sep='\t')

    data1 = data1.set_index('Hugo_Symbol', drop = True)
    data1 = data1.T
    data1.rename(index = lambda x : str(x).replace('-01', ''), inplace = True)
    data1 = data1.iloc[1:]
  
    data2 = data2.iloc[4:]
    data2 = data2[['Patient Identifier', 'Overall Survival Status', 'Overall Survival (Months)']]
    data2 = data2.set_index('Patient Identifier', drop = True)
  
    data = data2.merge(data1, left_on = 'Patient Identifier', right_index = True)

    return data

def calCorrFeature(data, name) :
    corr_mat = data.corr(method='pearson')
  
    upper_corr_mat = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
  
    unique_corr_pairs = upper_corr_mat.unstack().dropna()
  
    sorted_mat = abs(unique_corr_pairs).sort_values(ascending = False)

    choice = sorted_mat[:sort_num].index.to_numpy()
    corrFeature = np.array([])

    for i in range(len(choice)) :
        corrFeature = np.append(corrFeature, np.asarray(choice[i]).flatten())
  
    corrFeature = np.unique(corrFeature)
    corrFeature = np.append(corrFeature, 'e')
    corrFeature = np.append(corrFeature, 't')
  
    f = open("./" + name + "_corrFeature.txt", 'w')

    for index in corrFeature :
        f.write(index + '\t')
  
    f.close()
  
    return corrFeature

def selectCorrFeature(data, corrFeature) :
    corrFeature.drop(corrFeature.columns[-1], axis = 1, inplace = True)
    corrFeature = corrFeature.values.flatten()

    data = data[corrFeature]

    return data

def preprocessing (data) :
    missing_data = data.isnull().sum() / len(data) > drop_null

    missing_data = missing_data.to_frame()
    missing_data_index = missing_data.loc[missing_data[0] == True, :].index
    data = data.drop(missing_data_index, axis = 1)

    dup_cols = data.columns[data.columns.duplicated()]
    dup_indices = [ [i, col] for i, col in enumerate(data.columns) if col in dup_cols]

    dup_indices.sort(key=lambda x:x[1])
    dup_indices = list(list(zip(*dup_indices))[0])
    indexes = []

    for i in range(0, len(dup_indices), 2):
        pair = data.iloc[:, dup_indices[i:i + 2]]
        variances = pair.var()   
    if variances.iloc[0] < variances.iloc[1]:
        indexes.append(dup_indices[i])        
    else:
        indexes.append(dup_indices[i+1])

    data = data.iloc[:, [i for i in range(data.shape[1]) if i not in indexes]]
  
    data['Overall Survival Status'] = data['Overall Survival Status'].replace({'1:DECEASED': 1, '0:LIVING': 0})
    data = data.rename(columns={'Overall Survival (Months)': 't', 'Overall Survival Status': 'e'})
    data = data[data.t != '[Not Available]']

    return data