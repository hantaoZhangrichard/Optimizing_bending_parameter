
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from odbMaterial import *
from odbSection import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup() 
odb = session.openOdb(name="/Optimizing_bending_parameter/data/model/test0/simulation/Job-Model_base.odb")
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
    rpt_path = "/Optimizing_bending_parameter/data/model/test0/simulation/" + "strip_mises_" + step_name + ".rpt"

    with open(rpt_path, "w") as f:
        f.write("Node_ID    S_Mises    Orig.X      Orig.Y      Orig.Z \n")
        for v in output.values:
            element_id = v.elementLabel
            connected_nodes = strip_elements.elements[element_id-1].connectivity
            for node_id in connected_nodes:
                coordinates = strip_nodes.nodes[node_id-1].coordinates
                f.write(str(node_id) + " " + str(v.mises) + " " + str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2]) + "\n")

    f.close()

odb.close()