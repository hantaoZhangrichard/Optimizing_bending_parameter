import logging
import os
import subprocess
import sys
import shutil
import automation as at
import sys

SCRIPT_STR="""
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
"""


def build_spring_back(data_path, step=None, file_name="base"):
    inp_name = "Job-Model_springback"
    spring_back_name = "springback_" + file_name
    if step != None:
        # if analysis is performed step by step, just use the output of the last step
        model_name = 'Model_' + file_name + "_{}".format(step)
    else:
        model_name = 'Model_' + file_name
    recursion_path = data_path
    my_cae_path = os.path.join(recursion_path, "main.cae").replace("\\", "/")
    script_path = os.path.join(recursion_path, "script_spring_back.py").replace("\\", "/")
    with open(script_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(SCRIPT_STR)
        print('from abaqus import *')
        print('from abaqusConstants import *')
        print("from caeModules import *\nfrom driverUtils import executeOnCaeStartup")
        print("executeOnCaeStartup()")
        print('openMdb(pathName=\'%s\')' % (my_cae_path))
        # print('import os')    
        # print('os.chdir(r\"%s\")' % (recursion_path + "/simulation/"))
        print('mdb.Model(name=\'%s\', objectToCopy=mdb.models[\'%s\'])' % (spring_back_name, model_name))
        print('rp = mdb.models[\'%s\']' % (spring_back_name))
        print('del rp.parts[\'mould\']')
        print('del rp.parts[\'rotate\']')
        print('a = rp.rootAssembly')
        print(
            'a.deleteFeatures((\'translate\', \'rotate-z\', \'rotate-y\', \'rotate-x\', \'mould\', \'wire-z\', \'csys-z\', \'wire-y\', \'wire-z\', \'csys-y\', \'wire-x\', \'wire-x\', \'wire-y\', \'csys-x\', ))')
        print('rp.rootAssembly.deleteSets(setNames=(\'wire-x-Set-1\', \'wire-y-Set-1\', \'wire-z-Set-1\', ))')
        print('del rp.rootAssembly.surfaces[\'clamp_left\']')
        print('del rp.rootAssembly.sectionAssignments[2]')
        print('del rp.rootAssembly.sectionAssignments[1]')
        print('del rp.rootAssembly.sectionAssignments[0]')
        print('stepnum = len(rp.steps) - 1')
        print('for stepid in range(stepnum - 1, -1, -1):')
        print(' stepname = \'Step-\' + str(stepid)')
        print(' del rp.steps[stepname]')
        print('')
        print('')
        print('del rp.interactions[\'Int-1\']')
        print('del rp.interactionProperties[\'IntProp-1\']')
        print('del rp.constraints[\'constraint_clamp_strip\']')
        print('del rp.sections[\'ConnSect-Hinge\']')
        print(
            'rp.boundaryConditions.delete((\'mould\', \'rotate_x\', \'rotate_y\', \'rotate_z\', \'translate_fix_rotate\', \'translate_x\', \'translate_y\', \'translate_z\', ))')
        print(
            'rp.StaticStep(name=\'Step-1\', previous=\'Initial\', maxNumInc=10000, initialInc=1e-07, minInc=1e-10, nlgeom=ON)')
        print('instances=(rp.rootAssembly.instances[\'strip\'], )')
        print(
            'rp.InitialState(updateReferenceConfiguration=ON, fileName=\'Job-Model_base\', endStep=LAST_STEP, endIncrement=STEP_END, name=\'Predefined Field-1\', createStepName=\'Initial\', instances=instances)')

        print(
            'mdb.Job(name=\'%s\', model=\'%s\', description=\'\', memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine=\'\', scratch=\'\', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=16, numDomains=16, numGPUs=0)' % (
                inp_name, spring_back_name))
        print('mdb.jobs[\'%s\'].writeInput(consistencyChecking=OFF)' % (inp_name))
        print("mdb.jobs[\"%s\"].submit(consistencyChecking=OFF)" % (inp_name))
        print('mdb.save()')
        sys.stdout = old_stdout


def generate_springback_script(data_path: str, step=None):
    data_path = os.path.abspath(data_path)
    build_spring_back(data_path, step)
    p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", "noGUI=script_spring_back"],
        cwd=data_path,
        stderr=subprocess.PIPE, 
        stdout=subprocess.PIPE,
    )
    # noinspection DuplicatedCode
    try:
        outs, errs = p.communicate(timeout=300)
        print(outs.decode("utf-8"))
        print(errs.decode("utf-8"))
        logging.info("回弹脚本执行成功")
    except TimeoutError:
        p.kill()
        logging.info("回弹脚本运行失败，请检查代码")
        exit(-1)

mould_name = sys.argv[1]
step_name = sys.argv[2]

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    generate_springback_script("./data/model/" + mould_name + "/simulation/", step_name)
