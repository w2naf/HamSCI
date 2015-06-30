#!/usr/bin/env python
import sys 
import os 
import matplotlib
matplotlib.use
import pandas as pd
import numpy as np 

#specify filename (the file is in the same folder as the code 
fname='ds_CL.csv'

#Optional input path 
#(Caution: folder 'eclipse' does not exist yet!!!! make this folder and put file in it BEFORE using this code section!)
#input_path=os.path.join('data','eclipse')
#fpath=os.path.join(input,fname)

#Make data frame (make sure there is an index column that is seperate from your data unless part of the data is an index!) 
df_cl=pd.DataFrame.from_csv(fname)

import ipdb; ipdb.set_trace()
