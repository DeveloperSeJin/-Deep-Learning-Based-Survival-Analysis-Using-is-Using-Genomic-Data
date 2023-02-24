import numpy as np
import pandas as pd
import torch

def sort_data(path):
	''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
	Input:
		path: path to input dataset (which is expected to be a csv file).
	Output:
		x: sorted genomic inputs.
		ytime: sorted survival time (OS_MONTHS) corresponding to 'x'.
		yevent: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored.
		age: sorted age corresponding to 'x'.
	'''
	
	data = pd.read_csv(path)
	
	data.sort_values("t", ascending = False, inplace = True)
	
	x = data.drop(["t", "e"], axis = 1).values
	ytime = data.loc[:, ["t"]].values
	yevent = data.loc[:, ["e"]].values

	return(x, ytime, yevent)


def load_data(path, dtype):
	'''Load the sorted data, and then covert it to a Pytorch tensor.
	Input:
		path: path to input dataset (which is expected to be a csv file).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		X: a Pytorch tensor of 'x' from sort_data().
		YTIME: a Pytorch tensor of 'ytime' from sort_data().
		YEVENT: a Pytorch tensor of 'yevent' from sort_data().
		AGE: a Pytorch tensor of 'age' from sort_data().
	'''
	x, ytime, yevent = sort_data(path)

	X = torch.from_numpy(x).type(dtype)
	YTIME = torch.from_numpy(ytime).type(dtype)
	YEVENT = torch.from_numpy(yevent).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		X = X.cuda()
		YTIME = YTIME.cuda()
		YEVENT = YEVENT.cuda()
	###
	return(X, YTIME, YEVENT)
