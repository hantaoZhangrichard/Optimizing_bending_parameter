import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

def step_data_collection(start, end):
    for i in range(start, end+1):
        if i == 38 or i == 100:
            continue
        mould_name = "test" + str(i)
        cmd = ["python", "data_collection.py", mould_name]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
        p.wait(2400)
        if p.poll() == 0:
            print(cmd[1], mould_name, 'Success')
        else:
            print(cmd[1], 'Failure')

def build_dataset(start, end):
    dataset_X = pd.DataFrame()
    dataset_Y = pd.DataFrame()
    for i in range(start, end+1):
        mould_name = "test" + str(i)
        parameter_path = "/Xie_and_Zhang/data/mould_output/{}/".format(mould_name) + "param_base_rel.csv"
        if os.path.exists(parameter_path) == False:
            print("{} does not exist".format(mould_name))
            continue
        bending_parameter = pd.read_csv(parameter_path, header=None)
        # print(bending_parameter)
        step_check = True
        j = 0
        while step_check:
            step_name_x = "Step-" + str(j)
            step_name_y = "Step-" + str(j+1)
            csv_path_x = "/Xie_and_Zhang/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_x + ".csv"
            csv_path_y = "/Xie_and_Zhang/data/model/{}/simulation/".format(mould_name) + "strip_mises_" + step_name_y + ".csv"
            
            if os.path.exists(csv_path_x) and os.path.exists(csv_path_y):
                df_x = pd.read_csv(csv_path_x)
                df_y = pd.read_csv(csv_path_y)
                x = df_x["S_Mises"]
                # print(len(x))
                parameter = bending_parameter.iloc[j+1] - bending_parameter.iloc[j]
                parameter = parameter[parameter!=0]
                # print(parameter)
                # print(type(parameter))
                x = pd.concat([x, parameter]).reset_index(drop=True)
                # print(x)
                
                dataset_X = pd.concat([dataset_X, x], axis=1)
                dataset_Y = pd.concat([dataset_Y, df_y["S_Mises"]], axis=1)
                j += 1
            else:
                step_check = False
        # print(dataset_X)
        print("{} finished".format(mould_name))
    
    dataset_X.to_csv("/Xie_and_Zhang/dataset_X.csv", index=False, mode="w", header=False)
    dataset_Y.to_csv("/Xie_and_Zhang/dataset_Y.csv", index=False, mode="w", header=False)
    print("Created dataset csv")

if __name__ == "__main__":
    # step_data_collection(72, 100)
    build_dataset(5, 99)