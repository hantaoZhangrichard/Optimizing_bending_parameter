
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from odbMaterial import *
from odbSection import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup() 
odb = session.openOdb(name="/Optimizing_bending_parameter/data/model/test0/simulation/Job-Model_springback.odb")
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

rpt_path = "/Optimizing_bending_parameter/data/model/test0/simulation/" + "springback_output" + ".rpt"

with open(rpt_path, "w") as f:
    f.write("Node_ID    Springback \n")
    for v in output.values:
        f.write(str(v.nodeLabel) + " " + str(v.magnitude) + "\n")

f.close()

odb.close()