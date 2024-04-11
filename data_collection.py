import pandas as pd
import sys
import os
import subprocess

mould_name = sys.argv[1]
dir = "/Optimizing_bending_parameter/data/model/" + mould_name + "/simulation/"

package_script = '''
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from odbMaterial import *
from odbSection import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup() \n'''

stress_extraction_script = '''
num_step = len(odb.steps)
print num_step
for i in range(num_step):
    step_name = "Step-" + str(i)
    step = odb.steps[step_name]
    mises = step.frames[-1]
    mises_output = mises.fieldOutputs["S"]

    instance = odb.rootAssembly.instances["STRIP"]
    strip_nodes = instance.nodeSets['SET_ALL']

    # Mises are recorded on element object
    strip_elements = instance.elementSets['SET_ALL']
    # print len(strip.elements)


    output = mises_output.getSubset(region=strip_elements)
    output_values = output.values
    rpt_path = \"{}\" + "strip_mises_" + step_name + ".rpt"

    with open(rpt_path, "w") as f:
        f.write("Node_ID    S_Mises    Orig.X      Orig.Y      Orig.Z \\n")
        for v in output.values:
            element_id = v.elementLabel
            connected_nodes = strip_elements.elements[element_id-1].connectivity
            for node_id in connected_nodes:
                coordinates = strip_nodes.nodes[node_id-1].coordinates
                f.write(str(node_id) + " " + str(v.mises) + " " + str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2]) + "\\n")

    f.close()\n
'''

stress_extraction_script_2 = '''
step_name = "Step-" + str({step_num})
step = odb.steps[step_name]
mises = step.frames[-1]
mises_output = mises.fieldOutputs["S"]

instance = odb.rootAssembly.instances["STRIP"]
strip_nodes = instance.nodeSets['SET_ALL']

# Mises are recorded on element object
strip_elements = instance.elementSets['SET_ALL']
# print len(strip.elements)


output = mises_output.getSubset(region=strip_elements)
output_values = output.values
rpt_path = \"{}\" + "strip_mises_" + step_name + ".rpt"

with open(rpt_path, "w") as f:
    f.write("Node_ID    S_Mises    Orig.X      Orig.Y      Orig.Z \\n")
    for v in output.values:
        element_id = v.elementLabel
        connected_nodes = strip_elements.elements[element_id-1].connectivity
        for node_id in connected_nodes:
            coordinates = strip_nodes.nodes[node_id-1].coordinates
            f.write(str(node_id) + " " + str(v.mises) + " " + str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2]) + "\\n")

f.close()\n
'''

springback_extraction_script = '''
step_name = "Step-1"
step = odb.steps[step_name]
mises = step.frames[-1]
mises_output = mises.fieldOutputs["U"]
# print mises_output.values
springback = []
for v in mises_output.values:
    springback.append(v.magnitude)
# print max(springback)

instance = odb.rootAssembly.instances["STRIP"]
strip_nodes = instance.nodeSets['SET_ALL']
# Mises are recorded on element object
strip_elements = instance.elementSets['SET_ALL']

output = mises_output.getSubset(region=strip_nodes)
output_values = output.values

rpt_path = \"{}\" + "springback_output" + ".rpt"

with open(rpt_path, "w") as f:
    f.write("Node_ID    Springback \\n")
    for v in output.values:
        f.write(str(v.nodeLabel) + " " + str(v.magnitude) + "\\n")

f.close()\n
'''

def stress_collection_script(data_path, mould_name, step=None):
    script_path = os.path.join(data_path, "stress_collection_script.py")
    # print(script_path)
    if step == None:
        odb_path = os.path.join(data_path, "Job-Model_base.odb")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(package_script)
            f.write("odb = session.openOdb(name=\"{}\")".format(odb_path))
            f.write(stress_extraction_script.format(data_path))
            f.write("odb.close()")
    else:
        odb_path = os.path.join(data_path, "Job-Model_base_{}.odb".format(step))
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(package_script)
            f.write("odb = session.openOdb(name=\"{}\")".format(odb_path))
            f.write(stress_extraction_script_2.format(data_path, step_num = step))
            f.write("odb.close()")
    
    f.close()

    print("Successfully generated stress collection script for {}".format(mould_name))

    p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", f"noGUI={os.path.join(data_path, 'stress_collection_script')}"],
        cwd=data_path,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    p.wait(40)
    if p.poll() == 0:
        print('Stress collection success')
    else:
        print('Stress collection failure')
    rpt_to_csv(data_path)

def rpt_to_csv(data_path):
    step_check = True
    i = 0
    while step_check:
        step_name = "Step-" + str(i)
        rpt_path = data_path + "strip_mises_" + step_name + ".rpt"
        if os.path.exists(rpt_path):
            df = pd.read_csv(rpt_path, delim_whitespace=True)
            df = df.groupby(by="Node_ID").mean()
            df = df.sort_values(by="Node_ID")
            new_df = data_cleaning(df)
            print(len(new_df["S_Mises"]))
            if len(new_df["S_Mises"]) < 1512:
                print(df["S_Mises"])
            csv_path = rpt_path.replace(".rpt", ".csv")
            # print(csv_path)
            df.to_csv(csv_path)
            i += 1
        else:
            step_check = False
    rpt_path = data_path + "springback_output" + ".rpt"
    if os.path.exists(rpt_path):
            df = pd.read_csv(rpt_path, delim_whitespace=True)
            df = df.groupby(by="Node_ID").mean()
            df = df.sort_values(by="Node_ID")
            print(len(df["Springback"]))
            csv_path = rpt_path.replace(".rpt", ".csv")
            # print(csv_path)
            df.to_csv(csv_path)

    print ("data cleaning finished")

def data_cleaning(df):

    new_df = df[df["S_Mises"]<=300]
    new_df = df[df["S_Mises"]>=10]

    return new_df

def springback_collection_script(data_path):
    script_path = os.path.join(data_path, "springback_collection_script.py")
    # print(script_path)
    odb_path = os.path.join(data_path, "Job-Model_springback.odb")
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(package_script)
        f.write("odb = session.openOdb(name=\"{}\")".format(odb_path))
        f.write(springback_extraction_script.format(data_path))
        f.write("odb.close()")
    
    f.close()

    print("Successfully generated springback collection script for {}".format(mould_name))

    p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", f"noGUI={os.path.join(data_path, 'springback_collection_script')}"],
        cwd=data_path,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    p.wait(40)
    if p.poll() == 0:
        print('Springback collection success')
    else:
        print('Springback collection failure')

    rpt_to_csv(data_path)

if __name__ == "__main__":
    stress_collection_script(data_path=dir)
    
    springback_collection_script(dir)

    rpt_to_csv()

