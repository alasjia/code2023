import os 
import numpy as np
import pandas as pd

# path = "E:\TJRDTS_Jinan\onedaydata20220117.csv"
# raw_data = pd.read_csv(path, encoding = "utf-8")
# print("row number:",raw_data.shape[0])       #20672767
# print("column number:", raw_data.shape[1])   #16
# test_data = raw_data[:1000]
# test_data.to_csv("test_data1000.csv")

data = pd.read_csv("test_data1000.csv")
print(data)