import pandas as pd
from io import StringIO
file_path = 'strip_mises.rpt'
output_path_1 = "strip_force"
output_path_2 = "strip_coordinates"

def extract_step(file_path):
    with open(file_path, 'r') as f1:
        content = f1.readlines()
    for line in content:
        keyword_index = line.find("Step:")
        if keyword_index != -1:
            step_num = line[keyword_index + 6: keyword_index + 13]
            break
    return step_num

def extract_lines(file_path, output_path_1, output_path_2):
    coordinates = []
    mises = []
    start_marker1 = "S, Mises"
    start_marker2 = "STRIP"
    start_marker3 = "Coords"
    end_marker = "Attached elements"
    block_f = False
    block_s = False
    block_c = True

    with open(file_path, 'r') as f1:
        content = f1.readlines()
    for line in content:
        if end_marker in line:
            '''if "ASSEMBLY" in line:
                continue'''
            block_s = False
            block_c = False

        if start_marker1 in line:
            # print(line)
            block_f = True

        if start_marker3 in line:
            block_c = True
        if block_f:
            if start_marker2 in line:
                block_s = True

        if block_f and block_s and block_c:
            coordinates.append(line)

        if block_f and block_s and not block_c:
            mises.append(line)

    print(mises)
    with open(output_path_1, "w") as f2:
        for line in mises:
            f2.write(line)
    with open(output_path_2, "w") as f3:
        for line in coordinates:
            f3.write(line)
    f1.close()
    f2.close()
    f3.close()
    return step_num

def transfer_datatype(output_path, mode, type="Orig"):
    if mode == "coor":
        header = ["Type", "ID", "Orig_X", "Orig_Y", "Orig_Z", "Def_X", "Def_Y", "Def_Z"]
        df = pd.read_csv(output_path, names=header, delim_whitespace=True, header=None)
        coor_type = [type + "_X", type + "_Y", type + "_Z"]
        df.drop(columns=coor_type, inplace=True)
        df.drop(columns="Type", inplace=True)
    if mode == "force":
        header = ["Type", "ID", "Attached Elements", "S, Mises"]
        df = pd.read_csv(output_path, names=header, delim_whitespace=True, header=None)
        df = df.groupby(by="ID").mean()
    df = df.sort_values(by="ID")


    print(df)
    return df
    # print(df["ID"].dtypes)



if __name__ == "__main__":
    step_num = extract_step(file_path)
    output_path_1 = output_path_1 + "_" + step_num
    output_path_2 = output_path_2 + "_" + step_num
    extract_lines(file_path, output_path_1, output_path_2)
    coor_df = transfer_datatype(output_path_2, mode="coor")
    force_df = transfer_datatype(output_path_1, mode="force")
    coor_df.to_csv(output_path_2 + '.csv')
    force_df.to_csv(output_path_1 + '.csv')


