# encoding:utf-8
"""
    Abaqus 脚本中不变的部分
    注意：脚本字符串中需要适当空行，不要乱删，虽然有一些可能没用，但总之不要乱删
"""

HEADER_SCRIPT_STR = """# -*- coding: UTF-8 -*-
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
from connectorBehavior import *"""

# 如果存在这个model就删除
MODEL_SCRIPT_STR = """
if '{model_name}' in mdb.models: del mdb.models['{model_name}']

mdb.Model(modelType=STANDARD_EXPLICIT, name='{model_name}')
"""

CLAMP_SCRIPT_STR = """
# 建模方块
# 10 * 10可能有点小，调整为20 * 20试试
mdb.models['{model_name}'].ConstrainedSketch(name='__profile__', sheetSize=(3 * {half_length}))
mdb.models['{model_name}'].sketches['__profile__'].rectangle(
        point1=({half_length}, {half_length}), point2=(-{half_length}, -{half_length})
    )
mdb.models['{model_name}'].Part(dimensionality=THREE_D, name='rotate', type= DISCRETE_RIGID_SURFACE)
mdb.models['{model_name}'].parts['rotate'].BaseSolidExtrude(
        depth=4.0, sketch= mdb.models['{model_name}'].sketches['__profile__']
    )
del mdb.models['{model_name}'].sketches['__profile__']
# 六个面的中心参考点
mdb.models['{model_name}'].parts['rotate'].ReferencePoint(point=(0.0, 0.0, 2.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, {half_length}, 2.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=({half_length}, 0.0, 2.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, -{half_length}, 2.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=(-{half_length}, 0.0, 2.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, 0.0, 0.0))
mdb.models['{model_name}'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, 0.0, 4.0))
# 转换成壳，这里的cell list只有一个元素，直接用mask就好
mdb.models['{model_name}'].parts['rotate'].RemoveCells(cellList=
    mdb.models['{model_name}'].parts['rotate'].cells.getSequenceFromMask(mask=('[#1 ]', ), ))
# mesh
mdb.models['{model_name}'].parts['rotate'].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=20.0)
mdb.models['{model_name}'].parts['rotate'].generateMesh()
# 添加质量和转动惯量
mdb.models['{model_name}'].parts['rotate'].engineeringFeatures.PointMassInertia(
        alpha=0.0, composite=0.0, i11=1e-9, i22=1e-9, i33=1e-9, mass=1e-9,
        name='inertia', region=Region(
                vertices=mdb.models['{model_name}'].parts['rotate'].vertices[0:8],
                referencePoints=(mdb.models['{model_name}'].parts['rotate'].referencePoints[2],
            )
        )
    )
"""

STRIP_SCRIPT_STR = """
# 导入铝件截面草图
mdb.openStep('{strip_section_file}', scaleFromFile=OFF)
mdb.models['{model_name}'].ConstrainedSketchFromGeometryFile(geometryFile=mdb.acis, name='strip_section')
# 创建铝条
mdb.models['{model_name}'].ConstrainedSketch(name='__profile__', sheetSize=1500.0)
# mdb.models['{model_name}'].sketches['__profile__'].sketchOptions.setValues(
#     gridOrigin=(0.477320671081543, 1.32509183883667))
# 下面这个gridOrigin应该没影响才对
mdb.models['{model_name}'].sketches['__profile__'].sketchOptions.setValues(gridOrigin=(0, 0))
mdb.models['{model_name}'].sketches['__profile__'].retrieveSketch(
        sketch=mdb.models['{model_name}'].sketches['strip_section']
    )
mdb.models['{model_name}'].Part(dimensionality=THREE_D, name='strip', type=DEFORMABLE_BODY)
mdb.models['{model_name}'].parts['strip'].BaseSolidExtrude(
        depth={strip_length},
        sketch=mdb.models['{model_name}'].sketches['__profile__']
    )
del mdb.models['{model_name}'].sketches['__profile__']
# 创建铝条的表面集合
# 测试发现，铝条的倒数第1个面为z=0这个面，倒数第2个面为z=strip_length这个面
faces_count = len(mdb.models['{model_name}'].parts['strip'].faces)
# 除了两端的其他表面
mdb.models['{model_name}'].parts['strip'].Surface(name='surf_length', side1Faces=
    mdb.models['{model_name}'].parts['strip'].faces[0:(faces_count - 2)])
# 自由端，之后与夹钳绑定
mdb.models['{model_name}'].parts['strip'].Surface(name='surf_free', side1Faces=
    mdb.models['{model_name}'].parts['strip'].faces[(faces_count - 2):(faces_count - 1)])
# 固定端
mdb.models['{model_name}'].parts['strip'].Surface(name='surf_fixed', side1Faces=
    mdb.models['{model_name}'].parts['strip'].faces[(faces_count - 1):faces_count])
# 创建固定端集合
mdb.models['{model_name}'].parts['strip'].Set(
        faces=mdb.models['{model_name}'].parts['strip'].faces[(faces_count - 1):faces_count],
        name='set_fixed'
    )
# 创建全部铝条的集合，用于赋予材料属性
mdb.models['{model_name}'].parts['strip'].Set(
        cells=mdb.models['{model_name}'].parts['strip'].cells[0:1], name='set_all'
    )
"""

MOULD_SCRIPT_STR = """
# 导入模具
mdb.openStep('{mould_file}', scaleFromFile=OFF)
mdb.models['{model_name}'].PartFromGeometryFile(
        combine=True, dimensionality=THREE_D, geometryFile=mdb.acis, name='mould', type=DEFORMABLE_BODY
    )
# 转换成离散刚体
mdb.models['{model_name}'].parts['mould'].setValues(space=THREE_D, type=DISCRETE_RIGID_SURFACE)
# 添加参考点
mdb.models['{model_name}'].parts['mould'].ReferencePoint(point=mdb.models['{model_name}'].parts['mould'].vertices[0])
# 添加两个面
mdb.models['{model_name}'].parts['mould'].Surface(
        name='surf_down',
        side2Faces=mdb.models['{model_name}'].parts['mould'].faces[:]
    )
mdb.models['{model_name}'].parts['mould'].Surface(
        name='surf_up',
        side1Faces=mdb.models['{model_name}'].parts['mould'].faces[:]
    )
"""

MATERIAL_SCRIPT_STR = """
# 材料部分
# 创建材料
mdb.models['{model_name}'].Material(name='AL')
mdb.models['{model_name}'].materials['AL'].Density(table=(({density}, ), ))
mdb.models['{model_name}'].materials['AL'].Elastic(table=(({young}, {possion}), ))
mdb.models['{model_name}'].materials['AL'].Plastic(table={plastic_table})
mdb.models['{model_name}'].HomogeneousSolidSection(material='AL', name='Section-AL',thickness=None)
"""

DAMPING_SCRIPT_STR = """
# 这是一个可选项
mdb.models['{model_name}'].materials['AL'].Damping(alpha={alpha})
"""

ASSEMBLY_SCRIPT_STR = """
# step 1 为铝条赋予材料属性
mdb.models['{model_name}'].parts['strip'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=mdb.models['{model_name}'].parts['strip'].sets['set_all'],
        sectionName='Section-AL',
        thicknessAssignment=FROM_SECTION
    )

# step 2 构建夹钳模型
mdb.models['{model_name}'].rootAssembly.DatumCsysByDefault(CARTESIAN)
# 平移块
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON, name='translate', part=mdb.models['{model_name}'].parts['rotate']
    )
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('translate', ), vector=(0.0, 0.0, -2.0))
mdb.models['{model_name}'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 1.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('translate', ))
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('translate', ), vector=({X0} + {Dyz_Dxy_D}, {Y0}, {Z0}))
# z轴旋转
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON, name='rotate-z', part=mdb.models['{model_name}'].parts['rotate']
    )
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-z', ), vector=(0.0, 0.0, -2.0))
mdb.models['{model_name}'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 0.0, 1.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-z', ))
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-z', ), vector=({X0} + {Dyz_Dxy}, {Y0}, {Z0}))
# y轴旋转
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON, name='rotate-y', part=mdb.models['{model_name}'].parts['rotate']
    )
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-y', ), vector=(0.0, 0.0, -2.0))
mdb.models['{model_name}'].rootAssembly.rotate(angle=90.0, axisDirection=(1.0, 0.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-y', ))
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-y', ), vector=({X0} + {Dxy}, {Y0}, {Z0}))
# x轴旋转
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON, name='rotate-x', part=mdb.models['{model_name}'].parts['rotate']
    )
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-x', ), vector=(0.0, 0.0, -2.0))
mdb.models['{model_name}'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 1.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-x', ))
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('rotate-x', ), vector=({X0} + 2.0, {Y0}, {Z0}))

# z坐标系
mdb.models['{model_name}'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-z', origin=
    mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
    point1=mdb.models['{model_name}'].rootAssembly.instances['translate'].datums[6],
    point2=mdb.models['{model_name}'].rootAssembly.instances['translate'].datums[3])
# y坐标系
mdb.models['{model_name}'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-y', origin=
    mdb.models['{model_name}'].rootAssembly.instances['rotate-z'].referencePoints[2],
    point1=mdb.models['{model_name}'].rootAssembly.instances['rotate-z'].datums[4],
    point2=mdb.models['{model_name}'].rootAssembly.instances['rotate-z'].datums[8])
# x坐标系
mdb.models['{model_name}'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-x', origin=
    mdb.models['{model_name}'].rootAssembly.instances['rotate-y'].referencePoints[2],
    point1=mdb.models['{model_name}'].rootAssembly.instances['rotate-y'].datums[4],
    point2=mdb.models['{model_name}'].rootAssembly.instances['rotate-y'].datums[7])

# 连接器section
mdb.models['{model_name}'].ConnectorSection(assembledType=HINGE, name='ConnSect-Hinge')
mdb.models['{model_name}'].sections['ConnSect-Hinge'].setValues(behaviorOptions=(
    ConnectorElasticity(behavior=RIGID, table=(), independentComponents=(),
    components=(4, )), ))
mdb.models['{model_name}'].sections['ConnSect-Hinge'].behaviorOptions[0].ConnectorOptions(
    )
# z轴旋转
mdb.models['{model_name}'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
    mdb.models['{model_name}'].rootAssembly.instances['rotate-z'].referencePoints[2]),
    ))
mdb.models['{model_name}'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-z')
mdb.models['{model_name}'].rootAssembly.Set(edges=
    mdb.models['{model_name}'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-z-Set-1')
mdb.models['{model_name}'].rootAssembly.SectionAssignment(region=
    mdb.models['{model_name}'].rootAssembly.sets['wire-z-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['{model_name}'].rootAssembly.sectionAssignments[0].getSet()
mdb.models['{model_name}'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['{model_name}'].rootAssembly.datums[10], region=
    mdb.models['{model_name}'].rootAssembly.allSets['wire-z-Set-1'])
# y轴旋转
mdb.models['{model_name}'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['{model_name}'].rootAssembly.instances['rotate-z'].referencePoints[2],
    mdb.models['{model_name}'].rootAssembly.instances['rotate-y'].referencePoints[2]),
    ))
mdb.models['{model_name}'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-y')
mdb.models['{model_name}'].rootAssembly.Set(edges=
    mdb.models['{model_name}'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-y-Set-1')
mdb.models['{model_name}'].rootAssembly.SectionAssignment(region=
    mdb.models['{model_name}'].rootAssembly.sets['wire-y-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['{model_name}'].rootAssembly.sectionAssignments[1].getSet()
mdb.models['{model_name}'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['{model_name}'].rootAssembly.datums[11], region=
    mdb.models['{model_name}'].rootAssembly.allSets['wire-y-Set-1'])
# x轴旋转
mdb.models['{model_name}'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['{model_name}'].rootAssembly.instances['rotate-y'].referencePoints[2],
    mdb.models['{model_name}'].rootAssembly.instances['rotate-x'].referencePoints[2]),
    ))
mdb.models['{model_name}'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-x')
mdb.models['{model_name}'].rootAssembly.Set(edges=
    mdb.models['{model_name}'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-x-Set-1')
mdb.models['{model_name}'].rootAssembly.SectionAssignment(region=
    mdb.models['{model_name}'].rootAssembly.sets['wire-x-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['{model_name}'].rootAssembly.sectionAssignments[2].getSet()
mdb.models['{model_name}'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['{model_name}'].rootAssembly.datums[12], region=
    mdb.models['{model_name}'].rootAssembly.allSets['wire-x-Set-1'])

# step 3 导入铝条
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON,
        name='strip',
        part=mdb.models['{model_name}'].parts['strip'],
    )
mdb.models['{model_name}'].rootAssembly.rotate(
        angle=90.0,
        axisDirection=(0.0, 1.0, 0.0),
        axisPoint=(0.0, 0.0, 0.0),
        instanceList=('strip', )
    )

# step 4 导入模具
mdb.models['{model_name}'].rootAssembly.Instance(
        dependent=ON,
        name='mould',
        part=mdb.models['{model_name}'].parts['mould'],
    )
mdb.models['{model_name}'].rootAssembly.translate(instanceList=('mould', ), 
    vector=(0.0, -8.0, 0.0))

# step 5 设置铝条和模具的接触关系
# 接触属性
mdb.models['{model_name}'].ContactProperty('IntProp-1')
mdb.models['{model_name}'].interactionProperties['IntProp-1'].TangentialBehavior(
        dependencies=0,
        directionality=ISOTROPIC,
        elasticSlipStiffness=None,
        formulation=PENALTY,
        fraction=0.005,
        maximumElasticSlip=FRACTION,
        pressureDependency=OFF,
        shearStressLimit=None,
        slipRateDependency=OFF,
        table=(({friction_coeff}, ), ),
        temperatureDependency=OFF
    )
mdb.models['{model_name}'].interactionProperties['IntProp-1'].NormalBehavior(
        allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD
    )
# 接触
mdb.models['{model_name}'].ContactExp(createStepName='Initial', name='Int-1')
mdb.models['{model_name}'].interactions['Int-1'].includedPairs.setValuesInStep(
    addPairs=((
    mdb.models['{model_name}'].rootAssembly.instances['mould'].surfaces['surf_up'],
    mdb.models['{model_name}'].rootAssembly.instances['strip'].surfaces['surf_length']),
    ), stepName='Initial', useAllstar=OFF)
mdb.models['{model_name}'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
    assignments=((GLOBAL, SELF, 'IntProp-1'), ), stepName='Initial')
# 23/08/21 这里的面选择好像有些问题
mdb.models['{model_name}'].interactions['Int-1'].includedPairs.setValuesInStep(
    addPairs=((
    mdb.models['{model_name}'].rootAssembly.instances['mould'].surfaces['surf_down'], 
    mdb.models['{model_name}'].rootAssembly.instances['strip'].surfaces['surf_length']), 
    ), stepName='Initial')

# step 6 设置铝条和模具的绑定约束
# 建立夹钳表面
mdb.models['{model_name}'].rootAssembly.Surface(name='clamp_left', side1Faces=
    mdb.models['{model_name}'].rootAssembly.instances['rotate-x'].faces[5:6])
# 绑定夹钳和铝条
mdb.models['{model_name}'].Tie(
        adjust=ON,
        master=mdb.models['{model_name}'].rootAssembly.surfaces['clamp_left'],
        name='constraint_clamp_strip',
        positionToleranceMethod=COMPUTED,
        slave=mdb.models['{model_name}'].rootAssembly.instances['strip'].surfaces['surf_free'],
        thickness=ON,
        tieRotations=ON
    )

# step 7 初始化各个边界条件
# 固定模具
mdb.models['{model_name}'].EncastreBC(createStepName='Initial', localCsys=None,
    name='mould', region=Region(referencePoints=(
    mdb.models['{model_name}'].rootAssembly.instances['mould'].referencePoints[2],
    )))
# 固定铝条左边
mdb.models['{model_name}'].EncastreBC(createStepName='Initial', localCsys=None,
    name='strip_fixed', region=
    mdb.models['{model_name}'].rootAssembly.instances['strip'].sets['set_fixed'])
# 平移
############################
# mdb.models['{model_name}'].DisplacementBC(amplitude=UNSET, createStepName=
#     'Initial', distributionType=UNIFORM, fieldName='', localCsys=None, name=
#     'translate', region=Region(referencePoints=(
#     mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
#     )), u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET)
############################
# # 固定平移块的转动
mdb.models['{model_name}'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_fix_rotate',
        region=Region(referencePoints=(
                mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=UNSET,
        u2=UNSET,
        u3=UNSET,
        ur1=SET,
        ur2=SET,
        ur3=SET
    )
# translate_x
mdb.models['{model_name}'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_x',
        region=Region(referencePoints=(
                mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=SET,
        u2=UNSET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )
mdb.models['{model_name}'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_y',
        region=Region(referencePoints=(
                mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=UNSET,
        u2=SET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )
mdb.models['{model_name}'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_z',
        region=Region(referencePoints=(
                mdb.models['{model_name}'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=UNSET,
        u2=UNSET,
        u3=SET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )
######################################
# rotate_z
mdb.models['{model_name}'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_z', region=Region(
    edges=mdb.models['{model_name}'].rootAssembly.edges[2:3]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)
# rotate_y
mdb.models['{model_name}'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_y', region=Region(
    edges=mdb.models['{model_name}'].rootAssembly.edges[1:2]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)
# rotate_x
mdb.models['{model_name}'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_x', region=Region(
    edges=mdb.models['{model_name}'].rootAssembly.edges[0:1]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)
"""

JOB_SCRIPT_STR = """
mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF,description='', echoPrint=OFF,
    explicitPrecision=DOUBLE_PLUS_PACK,historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE,
    model='{model_name}',modelPrint=OFF, multiprocessingMode=DEFAULT, name='Job-{model_name}',
    nodalOutputPrecision=FULL, numCpus={cpu_num}, numDomains={cpu_num},parallelizationMethodExplicit=DOMAIN, 
    queue=None, resultsFormat=ODB,scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, 
    waitMinutes=0)
mdb.jobs['Job-{model_name}'].writeInput()

"""

FIRST_STEP_SCRIPT_STR = """
mdb.models['{model_name}'].ExplicitDynamicsStep(
    improvedDtMethod=ON,
    massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, {mass_scaling}, 0.0, None, 0, 0, 0.0, 0.0, 0, None), ),
    name='{step_name}',
    previous='{previous_step_name}',
    timePeriod={time_period})
"""

STEP_SCRIPT_STR = """
mdb.models['{model_name}'].ExplicitDynamicsStep(
    improvedDtMethod=ON,
    name='{step_name}',
    previous='{previous_step_name}',
    timePeriod={time_period})
"""

ADD_BC_SCRIPT_STR = """
mdb.models['{0}'].boundaryConditions['{1}'].setValuesInStep(amplitude='{2}', stepName='{3}', {4})
"""

TABULAR_AMP_SCRIPT_STR = """
mdb.models['{model_name}'].TabularAmplitude(data={amp_table}, name='{amp_name}', smooth=SOLVER_DEFAULT, timeSpan=STEP)
"""

SMOOTH_STEP_AMP_SCRIPT_STR = """
mdb.models['{model_name}'].SmoothStepAmplitude(data={amp_table}, name='{amp_name}', timeSpan=STEP)
"""

AMP_DICT = {
    "tabular": TABULAR_AMP_SCRIPT_STR,
    "smooth step": SMOOTH_STEP_AMP_SCRIPT_STR,
}
