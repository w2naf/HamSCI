#!/usr/bin/env python
import sys 
import os 
import matplotlib
matplotlib.use
import pandas as pd
import numpy as np 

#specify filename (the file is in the same folder as the code 
fname='ds_CL.csv'
#fpath="/home/km4ege/Downloads/ds_cl.txt"
df_cl=pd.DataFrame.from_csv(fname)

import ipdb; ipdb.set_trace()
