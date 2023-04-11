# /********* index.py ***********/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Load the Excel file
processData = pd.read_csv("../sudo_py_mac/data/mac_processes.csv")
# print(processData.columns)
# print(processData.dtypes)
objTypeCols = processData[[i for i in processData.columns if processData[i].dtype == 'object']]

#
corrprocessData = processData.corr()
print(corrprocessData)