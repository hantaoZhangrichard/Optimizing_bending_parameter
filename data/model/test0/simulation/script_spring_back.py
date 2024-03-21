
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

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
openMdb(pathName='C:/Optimizing_bending_parameter/data/model/test0/simulation/main.cae')
mdb.Model(name='springback_base', objectToCopy=mdb.models['Model_base'])
rp = mdb.models['springback_base']
del rp.parts['mould']
del rp.parts['rotate']
a = rp.rootAssembly
a.deleteFeatures(('translate', 'rotate-z', 'rotate-y', 'rotate-x', 'mould', 'wire-z', 'csys-z', 'wire-y', 'wire-z', 'csys-y', 'wire-x', 'wire-x', 'wire-y', 'csys-x', ))
rp.rootAssembly.deleteSets(setNames=('wire-x-Set-1', 'wire-y-Set-1', 'wire-z-Set-1', ))
del rp.rootAssembly.surfaces['clamp_left']
del rp.rootAssembly.sectionAssignments[2]
del rp.rootAssembly.sectionAssignments[1]
del rp.rootAssembly.sectionAssignments[0]
stepnum = len(rp.steps) - 1
for stepid in range(stepnum - 1, -1, -1):
 stepname = 'Step-' + str(stepid)
 del rp.steps[stepname]


del rp.interactions['Int-1']
del rp.interactionProperties['IntProp-1']
del rp.constraints['constraint_clamp_strip']
del rp.sections['ConnSect-Hinge']
rp.boundaryConditions.delete(('mould', 'rotate_x', 'rotate_y', 'rotate_z', 'translate_fix_rotate', 'translate_x', 'translate_y', 'translate_z', ))
rp.StaticStep(name='Step-1', previous='Initial', maxNumInc=10000, initialInc=1e-07, minInc=1e-10, nlgeom=ON)
instances=(rp.rootAssembly.instances['strip'], )
rp.InitialState(updateReferenceConfiguration=ON, fileName='Job-Model_base', endStep=LAST_STEP, endIncrement=STEP_END, name='Predefined Field-1', createStepName='Initial', instances=instances)
mdb.Job(name='Job-Model_springback', model='springback_base', description='', memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=16, numDomains=16, numGPUs=0)
mdb.jobs['Job-Model_springback'].writeInput(consistencyChecking=OFF)
mdb.jobs["Job-Model_springback"].submit(consistencyChecking=OFF)
mdb.save()
