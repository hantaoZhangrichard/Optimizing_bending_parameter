# -*- coding: UTF-8 -*-
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

if 'Model_base' in mdb.models: del mdb.models['Model_base']

mdb.Model(modelType=STANDARD_EXPLICIT, name='Model_base')

# 建模方块
# 10 * 10可能有点小，调整为20 * 20试试
mdb.models['Model_base'].ConstrainedSketch(name='__profile__', sheetSize=(3 * 5))
mdb.models['Model_base'].sketches['__profile__'].rectangle(
        point1=(5, 5), point2=(-5, -5)
    )
mdb.models['Model_base'].Part(dimensionality=THREE_D, name='rotate', type= DISCRETE_RIGID_SURFACE)
mdb.models['Model_base'].parts['rotate'].BaseSolidExtrude(
        depth=4.0, sketch= mdb.models['Model_base'].sketches['__profile__']
    )
del mdb.models['Model_base'].sketches['__profile__']
# 六个面的中心参考点
mdb.models['Model_base'].parts['rotate'].ReferencePoint(point=(0.0, 0.0, 2.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, 5, 2.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(5, 0.0, 2.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, -5, 2.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(-5, 0.0, 2.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, 0.0, 0.0))
mdb.models['Model_base'].parts['rotate'].DatumPointByCoordinate(coords=(0.0, 0.0, 4.0))
# 转换成壳，这里的cell list只有一个元素，直接用mask就好
mdb.models['Model_base'].parts['rotate'].RemoveCells(cellList=
    mdb.models['Model_base'].parts['rotate'].cells.getSequenceFromMask(mask=('[#1 ]', ), ))
# mesh
mdb.models['Model_base'].parts['rotate'].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=20.0)
mdb.models['Model_base'].parts['rotate'].generateMesh()
# 添加质量和转动惯量
mdb.models['Model_base'].parts['rotate'].engineeringFeatures.PointMassInertia(
        alpha=0.0, composite=0.0, i11=1e-9, i22=1e-9, i33=1e-9, mass=1e-9,
        name='inertia', region=Region(
                vertices=mdb.models['Model_base'].parts['rotate'].vertices[0:8],
                referencePoints=(mdb.models['Model_base'].parts['rotate'].referencePoints[2],
            )
        )
    )

# 导入模具
mdb.openStep('C:/Optimizing_bending_parameter/data/model/test0/mould.stp', scaleFromFile=OFF)
mdb.models['Model_base'].PartFromGeometryFile(
        combine=True, dimensionality=THREE_D, geometryFile=mdb.acis, name='mould', type=DEFORMABLE_BODY
    )
# 转换成离散刚体
mdb.models['Model_base'].parts['mould'].setValues(space=THREE_D, type=DISCRETE_RIGID_SURFACE)
# 添加参考点
mdb.models['Model_base'].parts['mould'].ReferencePoint(point=mdb.models['Model_base'].parts['mould'].vertices[0])
# 添加两个面
mdb.models['Model_base'].parts['mould'].Surface(
        name='surf_down',
        side2Faces=mdb.models['Model_base'].parts['mould'].faces[:]
    )
mdb.models['Model_base'].parts['mould'].Surface(
        name='surf_up',
        side1Faces=mdb.models['Model_base'].parts['mould'].faces[:]
    )

# 构建铝条
mdb.models['Model_base'].ConstrainedSketch(name='__profile__', sheetSize=100.0)
mdb.models['Model_base'].sketches['__profile__'].rectangle(point1=(-0.2, -2.0), 
    point2=(0, 2.0))
mdb.models['Model_base'].Part(dimensionality=THREE_D, name='strip', type=
    DEFORMABLE_BODY)
mdb.models['Model_base'].parts['strip'].BaseSolidExtrude(depth=50, sketch=
    mdb.models['Model_base'].sketches['__profile__'])
del mdb.models['Model_base'].sketches['__profile__']
# 创建铝条的表面集合
# 测试发现，铝条的倒数第1个面为z=0这个面，倒数第2个面为z=strip_length这个面
faces_count = len(mdb.models['Model_base'].parts['strip'].faces)
# 除了两端的其他表面
mdb.models['Model_base'].parts['strip'].Surface(name='surf_length', side1Faces=
    mdb.models['Model_base'].parts['strip'].faces[0:(faces_count - 2)])
# 自由端，之后与夹钳绑定
mdb.models['Model_base'].parts['strip'].Surface(name='surf_free', side1Faces=
    mdb.models['Model_base'].parts['strip'].faces[(faces_count - 2):(faces_count - 1)])
# 固定端
mdb.models['Model_base'].parts['strip'].Surface(name='surf_fixed', side1Faces=
    mdb.models['Model_base'].parts['strip'].faces[(faces_count - 1):faces_count])
# 创建固定端集合
mdb.models['Model_base'].parts['strip'].Set(
        faces=mdb.models['Model_base'].parts['strip'].faces[(faces_count - 1):faces_count],
        name='set_fixed'
    )
# 创建全部铝条的集合，用于赋予材料属性
mdb.models['Model_base'].parts['strip'].Set(
        cells=mdb.models['Model_base'].parts['strip'].cells[0:1], name='set_all'
    )

# 材料部分
# 创建材料
mdb.models['Model_base'].Material(name='AL')
mdb.models['Model_base'].materials['AL'].Density(table=((2.7e-09, ), ))
mdb.models['Model_base'].materials['AL'].Elastic(table=((69000.0, 0.37), ))
mdb.models['Model_base'].materials['AL'].Plastic(table=[(155.2387948, 0.0), (157.2580093, 0.000240422), (159.0641836, 0.000565474), (160.7413561, 0.000913373), (162.3659012, 0.001341927), (163.9012821, 0.001871773), (165.3514806, 0.00236361), (166.7479583, 0.002846528), (168.0804994, 0.003420002), (170.3955656, 0.003910109), (171.483574, 0.0044491), (172.5161607, 0.005127747), (173.5806354, 0.005904094), (174.5560144, 0.006701704), (175.4000434, 0.007511661), (176.3259781, 0.008318971), (177.1454839, 0.00912814), (177.8864309, 0.009869521), (178.6629142, 0.010727636), (179.4187407, 0.011614992), (180.1240757, 0.012591187), (180.8709855, 0.013565402), (181.5335261, 0.014442568), (182.1445537, 0.01532997), (182.8367231, 0.016283154), (183.4655953, 0.017256476), (184.0663523, 0.018219723), (184.6787899, 0.019240305), (185.2549637, 0.020133911), (185.7591619, 0.021077132), (186.3321057, 0.022163796), (186.8248449, 0.023183113), (187.2607698, 0.024261002), (187.6971593, 0.025337696), (188.0326923, 0.026386549), (188.3327999, 0.027464177), (188.5352856, 0.028388233), (188.6662972, 0.02937111), (188.727776, 0.030132647), (188.7496429, 0.030904199)])
mdb.models['Model_base'].HomogeneousSolidSection(material='AL', name='Section-AL',thickness=None)

# 这是一个可选项
mdb.models['Model_base'].materials['AL'].Damping(alpha=130)

mdb.models['Model_base'].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models['Model_base'].parts['strip'].edges.getSequenceFromMask((
    '[#20 ]', ), ), number=60)
mdb.models['Model_base'].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models['Model_base'].parts['strip'].edges.getSequenceFromMask((
    '[#10 ]', ), ), number=2)
mdb.models['Model_base'].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models['Model_base'].parts['strip'].edges.getSequenceFromMask((
    '[#80 ]', ), ), number=6)
mdb.models['Model_base'].parts['strip'].generateMesh()

mdb.models['Model_base'].parts['mould'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=1)
mdb.models['Model_base'].parts['mould'].generateMesh()

# step 1 为铝条赋予材料属性
mdb.models['Model_base'].parts['strip'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=mdb.models['Model_base'].parts['strip'].sets['set_all'],
        sectionName='Section-AL',
        thicknessAssignment=FROM_SECTION
    )

# step 2 构建夹钳模型
mdb.models['Model_base'].rootAssembly.DatumCsysByDefault(CARTESIAN)
# 平移块
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON, name='translate', part=mdb.models['Model_base'].parts['rotate']
    )
mdb.models['Model_base'].rootAssembly.translate(instanceList=('translate', ), vector=(0.0, 0.0, -2.0))
mdb.models['Model_base'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 1.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('translate', ))
mdb.models['Model_base'].rootAssembly.translate(instanceList=('translate', ), vector=(50 + 321, 0, 0))
# z轴旋转
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON, name='rotate-z', part=mdb.models['Model_base'].parts['rotate']
    )
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-z', ), vector=(0.0, 0.0, -2.0))
mdb.models['Model_base'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 0.0, 1.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-z', ))
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-z', ), vector=(50 + 281, 0, 0))
# y轴旋转
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON, name='rotate-y', part=mdb.models['Model_base'].parts['rotate']
    )
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-y', ), vector=(0.0, 0.0, -2.0))
mdb.models['Model_base'].rootAssembly.rotate(angle=90.0, axisDirection=(1.0, 0.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-y', ))
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-y', ), vector=(50 + 205, 0, 0))
# x轴旋转
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON, name='rotate-x', part=mdb.models['Model_base'].parts['rotate']
    )
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-x', ), vector=(0.0, 0.0, -2.0))
mdb.models['Model_base'].rootAssembly.rotate(angle=90.0, axisDirection=(0.0, 1.0, 0.0),
    axisPoint=(0.0, 0.0, 0.0), instanceList=('rotate-x', ))
mdb.models['Model_base'].rootAssembly.translate(instanceList=('rotate-x', ), vector=(50 + 2.0, 0, 0))

# z坐标系
mdb.models['Model_base'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-z', origin=
    mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
    point1=mdb.models['Model_base'].rootAssembly.instances['translate'].datums[6],
    point2=mdb.models['Model_base'].rootAssembly.instances['translate'].datums[3])
# y坐标系
mdb.models['Model_base'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-y', origin=
    mdb.models['Model_base'].rootAssembly.instances['rotate-z'].referencePoints[2],
    point1=mdb.models['Model_base'].rootAssembly.instances['rotate-z'].datums[4],
    point2=mdb.models['Model_base'].rootAssembly.instances['rotate-z'].datums[8])
# x坐标系
mdb.models['Model_base'].rootAssembly.DatumCsysByThreePoints(coordSysType=CARTESIAN,
    name='csys-x', origin=
    mdb.models['Model_base'].rootAssembly.instances['rotate-y'].referencePoints[2],
    point1=mdb.models['Model_base'].rootAssembly.instances['rotate-y'].datums[4],
    point2=mdb.models['Model_base'].rootAssembly.instances['rotate-y'].datums[7])

# 连接器section
mdb.models['Model_base'].ConnectorSection(assembledType=HINGE, name='ConnSect-Hinge')
mdb.models['Model_base'].sections['ConnSect-Hinge'].setValues(behaviorOptions=(
    ConnectorElasticity(behavior=RIGID, table=(), independentComponents=(),
    components=(4, )), ))
mdb.models['Model_base'].sections['ConnSect-Hinge'].behaviorOptions[0].ConnectorOptions(
    )
# z轴旋转
mdb.models['Model_base'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
    mdb.models['Model_base'].rootAssembly.instances['rotate-z'].referencePoints[2]),
    ))
mdb.models['Model_base'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-z')
mdb.models['Model_base'].rootAssembly.Set(edges=
    mdb.models['Model_base'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-z-Set-1')
mdb.models['Model_base'].rootAssembly.SectionAssignment(region=
    mdb.models['Model_base'].rootAssembly.sets['wire-z-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['Model_base'].rootAssembly.sectionAssignments[0].getSet()
mdb.models['Model_base'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['Model_base'].rootAssembly.datums[10], region=
    mdb.models['Model_base'].rootAssembly.allSets['wire-z-Set-1'])
# y轴旋转
mdb.models['Model_base'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['Model_base'].rootAssembly.instances['rotate-z'].referencePoints[2],
    mdb.models['Model_base'].rootAssembly.instances['rotate-y'].referencePoints[2]),
    ))
mdb.models['Model_base'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-y')
mdb.models['Model_base'].rootAssembly.Set(edges=
    mdb.models['Model_base'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-y-Set-1')
mdb.models['Model_base'].rootAssembly.SectionAssignment(region=
    mdb.models['Model_base'].rootAssembly.sets['wire-y-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['Model_base'].rootAssembly.sectionAssignments[1].getSet()
mdb.models['Model_base'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['Model_base'].rootAssembly.datums[11], region=
    mdb.models['Model_base'].rootAssembly.allSets['wire-y-Set-1'])
# x轴旋转
mdb.models['Model_base'].rootAssembly.WirePolyLine(mergeType=IMPRINT, meshable=False,
    points=((
    mdb.models['Model_base'].rootAssembly.instances['rotate-y'].referencePoints[2],
    mdb.models['Model_base'].rootAssembly.instances['rotate-x'].referencePoints[2]),
    ))
mdb.models['Model_base'].rootAssembly.features.changeKey(fromName='Wire-1', toName=
    'wire-x')
mdb.models['Model_base'].rootAssembly.Set(edges=
    mdb.models['Model_base'].rootAssembly.edges.getSequenceFromMask(('[#1 ]', ), ),
    name='wire-x-Set-1')
mdb.models['Model_base'].rootAssembly.SectionAssignment(region=
    mdb.models['Model_base'].rootAssembly.sets['wire-x-Set-1'], sectionName=
    'ConnSect-Hinge')
mdb.models['Model_base'].rootAssembly.sectionAssignments[2].getSet()
mdb.models['Model_base'].rootAssembly.ConnectorOrientation(localCsys1=
    mdb.models['Model_base'].rootAssembly.datums[12], region=
    mdb.models['Model_base'].rootAssembly.allSets['wire-x-Set-1'])

# step 3 导入铝条
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON,
        name='strip',
        part=mdb.models['Model_base'].parts['strip'],
    )
mdb.models['Model_base'].rootAssembly.rotate(
        angle=90.0,
        axisDirection=(0.0, 1.0, 0.0),
        axisPoint=(0.0, 0.0, 0.0),
        instanceList=('strip', )
    )

# step 4 导入模具
mdb.models['Model_base'].rootAssembly.Instance(
        dependent=ON,
        name='mould',
        part=mdb.models['Model_base'].parts['mould'],
    )
mdb.models['Model_base'].rootAssembly.translate(instanceList=('mould', ), 
    vector=(0.0, -8.0, 0.0))

# step 5 设置铝条和模具的接触关系
# 接触属性
mdb.models['Model_base'].ContactProperty('IntProp-1')
mdb.models['Model_base'].interactionProperties['IntProp-1'].TangentialBehavior(
        dependencies=0,
        directionality=ISOTROPIC,
        elasticSlipStiffness=None,
        formulation=PENALTY,
        fraction=0.005,
        maximumElasticSlip=FRACTION,
        pressureDependency=OFF,
        shearStressLimit=None,
        slipRateDependency=OFF,
        table=((0.1, ), ),
        temperatureDependency=OFF
    )
mdb.models['Model_base'].interactionProperties['IntProp-1'].NormalBehavior(
        allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD
    )
# 接触
mdb.models['Model_base'].ContactExp(createStepName='Initial', name='Int-1')
mdb.models['Model_base'].interactions['Int-1'].includedPairs.setValuesInStep(
    addPairs=((
    mdb.models['Model_base'].rootAssembly.instances['mould'].surfaces['surf_up'],
    mdb.models['Model_base'].rootAssembly.instances['strip'].surfaces['surf_length']),
    ), stepName='Initial', useAllstar=OFF)
mdb.models['Model_base'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
    assignments=((GLOBAL, SELF, 'IntProp-1'), ), stepName='Initial')
# 23/08/21 这里的面选择好像有些问题
mdb.models['Model_base'].interactions['Int-1'].includedPairs.setValuesInStep(
    addPairs=((
    mdb.models['Model_base'].rootAssembly.instances['mould'].surfaces['surf_down'], 
    mdb.models['Model_base'].rootAssembly.instances['strip'].surfaces['surf_length']), 
    ), stepName='Initial')

# step 6 设置铝条和模具的绑定约束
# 建立夹钳表面
mdb.models['Model_base'].rootAssembly.Surface(name='clamp_left', side1Faces=
    mdb.models['Model_base'].rootAssembly.instances['rotate-x'].faces[5:6])
# 绑定夹钳和铝条
mdb.models['Model_base'].Tie(
        adjust=ON,
        master=mdb.models['Model_base'].rootAssembly.surfaces['clamp_left'],
        name='constraint_clamp_strip',
        positionToleranceMethod=COMPUTED,
        slave=mdb.models['Model_base'].rootAssembly.instances['strip'].surfaces['surf_free'],
        thickness=ON,
        tieRotations=ON
    )

# step 7 初始化各个边界条件
# 固定模具
mdb.models['Model_base'].EncastreBC(createStepName='Initial', localCsys=None,
    name='mould', region=Region(referencePoints=(
    mdb.models['Model_base'].rootAssembly.instances['mould'].referencePoints[2],
    )))
# 固定铝条左边
mdb.models['Model_base'].EncastreBC(createStepName='Initial', localCsys=None,
    name='strip_fixed', region=
    mdb.models['Model_base'].rootAssembly.instances['strip'].sets['set_fixed'])
# 平移
############################
# mdb.models['Model_base'].DisplacementBC(amplitude=UNSET, createStepName=
#     'Initial', distributionType=UNIFORM, fieldName='', localCsys=None, name=
#     'translate', region=Region(referencePoints=(
#     mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
#     )), u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET)
############################
# # 固定平移块的转动
mdb.models['Model_base'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_fix_rotate',
        region=Region(referencePoints=(
                mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=UNSET,
        u2=UNSET,
        u3=UNSET,
        ur1=SET,
        ur2=SET,
        ur3=SET
    )
# translate_x
mdb.models['Model_base'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_x',
        region=Region(referencePoints=(
                mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=SET,
        u2=UNSET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )
mdb.models['Model_base'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_y',
        region=Region(referencePoints=(
                mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
            )),
        u1=UNSET,
        u2=SET,
        u3=UNSET,
        ur1=UNSET,
        ur2=UNSET,
        ur3=UNSET
    )
mdb.models['Model_base'].DisplacementBC(
        amplitude=UNSET,
        createStepName='Initial',
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
        name='translate_z',
        region=Region(referencePoints=(
                mdb.models['Model_base'].rootAssembly.instances['translate'].referencePoints[2],
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
mdb.models['Model_base'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_z', region=Region(
    edges=mdb.models['Model_base'].rootAssembly.edges[2:3]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)
# rotate_y
mdb.models['Model_base'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_y', region=Region(
    edges=mdb.models['Model_base'].rootAssembly.edges[1:2]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)
# rotate_x
mdb.models['Model_base'].ConnDisplacementBC(amplitude=UNSET, createStepName=
    'Initial', distributionType=UNIFORM, name='rotate_x', region=Region(
    edges=mdb.models['Model_base'].rootAssembly.edges[0:1]), u1=UNSET, u2=UNSET
    , u3=UNSET, ur1=SET, ur2=UNSET, ur3=UNSET)

mdb.models['Model_base'].SmoothStepAmplitude(data=((0.0, 0.0), (0.1, 1.0)), name='amp_basic', timeSpan=STEP)

mdb.models['Model_base'].ExplicitDynamicsStep(
    improvedDtMethod=ON,
    massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 200, 0.0, None, 0, 0, 0.0, 0.0, 0, None), ),
    name='Step-0',
    previous='Initial',
    timePeriod=0.1)

mdb.models['Model_base'].boundaryConditions['translate_fix_rotate'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', ur1=0, ur2=0, ur3=0)

mdb.models['Model_base'].boundaryConditions['translate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', u1=0.1)

mdb.models['Model_base'].boundaryConditions['translate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', u2=0)

mdb.models['Model_base'].boundaryConditions['translate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', u3=0)

mdb.models['Model_base'].boundaryConditions['rotate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', ur1=0)

mdb.models['Model_base'].boundaryConditions['rotate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', ur1=0)

mdb.models['Model_base'].boundaryConditions['rotate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-0', ur1=0)

mdb.models['Model_base'].ExplicitDynamicsStep(
    improvedDtMethod=ON,
    name='Step-1',
    previous='Step-0',
    timePeriod=0.1)

mdb.models['Model_base'].boundaryConditions['translate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', u1=-0.1521150846325554)

mdb.models['Model_base'].boundaryConditions['translate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', u2=11.984517959837767)

mdb.models['Model_base'].boundaryConditions['translate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', u3=0.0)

mdb.models['Model_base'].boundaryConditions['rotate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', ur1=0.03903410176695613)

mdb.models['Model_base'].boundaryConditions['rotate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', ur1=0.0)

mdb.models['Model_base'].boundaryConditions['rotate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-1', ur1=0.0)

mdb.models['Model_base'].ExplicitDynamicsStep(
    improvedDtMethod=ON,
    name='Step-2',
    previous='Step-1',
    timePeriod=0.1)

mdb.models['Model_base'].boundaryConditions['translate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', u1=-0.19125764149265478)

mdb.models['Model_base'].boundaryConditions['translate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', u2=4.746996046981309)

mdb.models['Model_base'].boundaryConditions['translate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', u3=0.0)

mdb.models['Model_base'].boundaryConditions['rotate_z'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', ur1=0.016519475424908704)

mdb.models['Model_base'].boundaryConditions['rotate_y'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', ur1=0.0)

mdb.models['Model_base'].boundaryConditions['rotate_x'].setValuesInStep(amplitude='amp_basic', stepName='Step-2', ur1=0.0)

mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF,description='', echoPrint=OFF,
    explicitPrecision=DOUBLE_PLUS_PACK,historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE,
    model='Model_base',modelPrint=OFF, multiprocessingMode=DEFAULT, name='Job-Model_base',
    nodalOutputPrecision=FULL, numCpus=16, numDomains=16,parallelizationMethodExplicit=DOMAIN, 
    queue=None, resultsFormat=ODB,scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, 
    waitMinutes=0)
mdb.jobs['Job-Model_base'].writeInput()

del mdb.models['Model-1']
mdb.saveAs(pathName="C:/Optimizing_bending_parameter/data/model/test0/simulation/main.cae")

executeOnCaeStartup()
openMdb("C:/Optimizing_bending_parameter/data/model/test0/simulation/main.cae")
mdb.jobs['Job-Model_base'].submit(consistencyChecking=OFF)
mdb.save()

