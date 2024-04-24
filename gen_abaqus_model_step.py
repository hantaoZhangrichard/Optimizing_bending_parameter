import os
import json
import subprocess
import automation as at
import sys

from core.create_model_script_generator import CreateModelScriptGenerator
from core.param_util import calc_param_list

# from . import mesh_script

# from ...common.default_config import DEFAULT_CONFIG
# from ...common.config_tools import read_config

# from ._default_scripts import STRIP_SCRIPT_STR
# from .gen_section_script import gen_section_script

# 这里应该不用改
DEFAULT_CONFIG = {
    "model_name": "Model_base",
    "new_model": True,
    "step_config": {
        "has_pre_length": True,  # 暂时可能没啥用
        "global_time_period": 0.1,
        "basic_amp_table": ((0.0, 0.0), (0.1, 1.0)),  # 默认的amp曲线
        "amp_type": "smooth step",
        "only_use_amp_table": False,
    },
    "material": {
        "density": 2.7e-09,
        "young": 69000.0,
        "possion": 0.37,
        "has_damping": True,
        "alpha": 130,
        "plastic_str": "[(155.2387948, 0.0), (157.2580093, 0.000240422), (159.0641836, 0.000565474), (160.7413561, 0.000913373), (162.3659012, 0.001341927), (163.9012821, 0.001871773), (165.3514806, 0.00236361), (166.7479583, 0.002846528), (168.0804994, 0.003420002), (170.3955656, 0.003910109), (171.483574, 0.0044491), (172.5161607, 0.005127747), (173.5806354, 0.005904094), (174.5560144, 0.006701704), (175.4000434, 0.007511661), (176.3259781, 0.008318971), (177.1454839, 0.00912814), (177.8864309, 0.009869521), (178.6629142, 0.010727636), (179.4187407, 0.011614992), (180.1240757, 0.012591187), (180.8709855, 0.013565402), (181.5335261, 0.014442568), (182.1445537, 0.01532997), (182.8367231, 0.016283154), (183.4655953, 0.017256476), (184.0663523, 0.018219723), (184.6787899, 0.019240305), (185.2549637, 0.020133911), (185.7591619, 0.021077132), (186.3321057, 0.022163796), (186.8248449, 0.023183113), (187.2607698, 0.024261002), (187.6971593, 0.025337696), (188.0326923, 0.026386549), (188.3327999, 0.027464177), (188.5352856, 0.028388233), (188.6662972, 0.02937111), (188.727776, 0.030132647), (188.7496429, 0.030904199)]"
    },
    "friction_coeff": 0.1,
    "mould_file": "mould.stp",
    "simulation_root": "simulation",
    "cae_name": "main",
    "mass_scaling": 200,
    "clamp_half_length": 5, #0.8
    "cpu_num": 16,

    # # 下面两个配置就是方便调整idx用的
    # "idx_mode": "default",
    # "idx_config": None,
    # "max_step_dis": 15,  # 默认的idx的限制
    # "resample": True,  # 是否做重采样
}

STRIP_SCRIPT = """
# 构建铝条
mdb.models["{model_name}"].ConstrainedSketch(name='__profile__', sheetSize=100.0)
mdb.models["{model_name}"].sketches['__profile__'].rectangle(point1={}, 
    point2={})
mdb.models["{model_name}"].Part(dimensionality=THREE_D, name='strip', type=
    DEFORMABLE_BODY)
mdb.models["{model_name}"].parts['strip'].BaseSolidExtrude(depth={}, sketch=
    mdb.models["{model_name}"].sketches['__profile__'])
del mdb.models["{model_name}"].sketches['__profile__']
# 创建铝条的表面集合
# 测试发现，铝条的倒数第1个面为z=0这个面，倒数第2个面为z=strip_length这个面
faces_count = len(mdb.models["{model_name}"].parts['strip'].faces)
# 除了两端的其他表面
mdb.models["{model_name}"].parts['strip'].Surface(name='surf_length', side1Faces=
    mdb.models["{model_name}"].parts['strip'].faces[0:(faces_count - 2)])
# 自由端，之后与夹钳绑定
mdb.models["{model_name}"].parts['strip'].Surface(name='surf_free', side1Faces=
    mdb.models["{model_name}"].parts['strip'].faces[(faces_count - 2):(faces_count - 1)])
# 固定端
mdb.models["{model_name}"].parts['strip'].Surface(name='surf_fixed', side1Faces=
    mdb.models["{model_name}"].parts['strip'].faces[(faces_count - 1):faces_count])
# 创建固定端集合
mdb.models["{model_name}"].parts['strip'].Set(
        faces=mdb.models["{model_name}"].parts['strip'].faces[(faces_count - 1):faces_count],
        name='set_fixed'
    )
# 创建全部铝条的集合，用于赋予材料属性
mdb.models["{model_name}"].parts['strip'].Set(
        cells=mdb.models["{model_name}"].parts['strip'].cells[0:1], name='set_all'
    )
"""

MESH_SCRIPT = """
mdb.models["{model_name}"].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models["{model_name}"].parts['strip'].edges.getSequenceFromMask((
    '[#20 ]', ), ), number={})
mdb.models["{model_name}"].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models["{model_name}"].parts['strip'].edges.getSequenceFromMask((
    '[#10 ]', ), ), number={})
mdb.models["{model_name}"].parts['strip'].seedEdgeByNumber(constraint=FINER, 
    edges=mdb.models["{model_name}"].parts['strip'].edges.getSequenceFromMask((
    '[#80 ]', ), ), number={})
mdb.models["{model_name}"].parts['strip'].generateMesh()

mdb.models["{model_name}"].parts['mould'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=1)
mdb.models["{model_name}"].parts['mould'].generateMesh()
"""


class CustomCreateModelScriptGenerator(CreateModelScriptGenerator):
    def __init__(self, step=0):
        super().__init__()
        self.gen_strip_section_script = None
        self.mesh_script = None
        self.step = step

    def gen_all(self, data_path):
        self._script_io_wrapper = open(self.config["output_file"], "w", encoding="utf-8")
        self._gen_header_script()
        self._gen_clamp_script()
        self._gen_mould_script()
        self._gen_strip_scrip()
        self._gen_material_script()
        self._gen_mesh()
        self._gen_assembly_script()
        self._gen_step_script()
        self._script_io_wrapper.write("os.chdir('{}')\n".format((data_path+"/simulation")).replace("\\", "/"))
        self._gen_job_script()
        self._gen_save_model()
        self._gen_submit_job()
        self._script_io_wrapper.close()
        self._script_io_wrapper = None
    
    def gen_one_step(self, data_path):
        # print(self.step)
        self._script_io_wrapper = open(self.config["output_file"], "w", encoding="utf-8")
        self._gen_header_script()
        self._script_io_wrapper.write("executeOnCaeStartup()\n")
        self._script_io_wrapper.write("openMdb(\"{}\")\n".format(self.config["cae_path"]))
        self._script_io_wrapper.write("os.chdir('{}')\n".format((data_path+"/simulation")).replace("\\", "/"))
        self._script_io_wrapper.write("mdb.Model(name='Model_base_{}', objectToCopy=mdb.models['Model_base_{}'])\n".format(str(self.step), str(self.step-1)))
        self._script_io_wrapper.write(
            "mdb.models['Model_base_{}'].setValues(restartJob='Job-Model_base_{}', restartStep='Step-{}')\n".format(
                str(self.step), str(self.step-1), str(self.step-1)))
        self._gen_v1(self.step)
        job_script = """mdb.Job(name='Job-Model_base_{step_name}', model='{model_name}', description='', type=RESTART, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
        nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
        contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
        resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains={cpu_num}, 
        activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus={cpu_num})\nmdb.jobs['Job-Model_base_{step_name}'].writeInput()\n"""
        self._script_io_wrapper.write(job_script.format(
            step_name = str(self.step),
            model_name = self.config["model_name"],
            cpu_num = self.config["cpu_num"],
        ))
        self._gen_save_model()
        job_name = "Job-Model_base_{}".format(self.step)
        self._gen_submit_job(job_name)




    def set_gen_strip_section_script(self, script: str):
        self.gen_strip_section_script = script

    def _gen_strip_scrip(self):
        self._check_config("model_name", "strip_length")
        self._script_io_wrapper.write(STRIP_SCRIPT.format(
            
            self.config["strip_section_shape"][0],
            self.config["strip_section_shape"][1],
            self.config["strip_length"],
            model_name = self.config["model_name"],
        ))

    def _gen_mesh(self):
        self._script_io_wrapper.write(MESH_SCRIPT.format(
            
            *self.config["mesh_num"],
            model_name = self.config["model_name"],
        ))


def gen_abaqus_model(data_path: str, user_config: dict, step: int):
    # 生成 config
    static_path = os.path.abspath(data_path)
    recursion_path = os.path.abspath(data_path)
    data_path = os.path.abspath(data_path)

    step_name = "Step-" + str(step)

    print("开始生成仿真模型")
    print("正在读取配置")
    _config = DEFAULT_CONFIG
    _config.update({"output_file": os.path.join(recursion_path, "script_create_model_{}.py".format(step_name))})
    _config.update({"model_name": "Model_base_" + str(step)})
    # 先读取文件中的config
    _config.update(user_config)
    _config["cae_path"] = os.path.join(
        os.path.abspath(recursion_path),
        _config["simulation_root"],
        _config["cae_name"] + ".cae",
    ).replace("\\", "/")
    _config["mould_file"] = os.path.join(
        os.path.abspath(recursion_path),
        _config["mould_file"],
    ).replace("\\", "/")
    with open(os.path.join(recursion_path, "config_lock.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(_config, indent=4, ensure_ascii=False))

    # 初始化仿真路径
    simulation_root = os.path.abspath(os.path.join(recursion_path, _config["simulation_root"]))
    os.makedirs(simulation_root, exist_ok=True)
    print("static 文件夹绝对路径：" + static_path)
    print("recursion 文件夹绝对路径：" + recursion_path)
    print("cae 文件路径：" + _config["cae_path"])

    # # 计算夹钳参数
    # param_list = calc_param_list(
    #     recursion_path=recursion_path,
    #     strip_length=_config["strip_length"],
    #     pre_length=_config["pre_length"],
    #     k=_config["k"],
    #     idx_mode=_config["idx_mode"],
    #     idx_config=_config["idx_config"],
    #     max_step_dis=_config["max_step_dis"],
    #     config=_config,
    # )
    # print(param_list)
    # if _config.get("just_param", False):
    #     return
    # _config["param_list"] = param_list
    # print(_config)

    if step_name == "Step-0":
        model_generator = CustomCreateModelScriptGenerator()
        model_generator.set_config(_config)

        model_generator.gen_all(data_path)
    else:
        model_generator = CustomCreateModelScriptGenerator(step)
        model_generator.set_config(_config)
        model_generator.gen_one_step(data_path)

    print("生成abaqus建模脚本")
    print("运行abaqus建模脚本")
    print("abaqus建模脚本位置：" + f"{os.path.join(recursion_path, 'script_create_model_{}'.format(step_name))}")
    print("abaqus 工作路径：" + os.path.join(recursion_path, _config["simulation_root"]))
    '''
    p = subprocess.Popen(
        ["cmd", "/c", "abaqus", "cae", f"noGUI={os.path.join(recursion_path, 'script_create_model_{}'.format(step_name))}"],
        cwd=os.path.join(recursion_path, _config["simulation_root"]),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        outs, errs = p.communicate(timeout=2400)
        print(outs)
        print(errs)
        # print(outs.decode("utf-8"))
        # print(errs.decode("utf-8"))
        print("abaqus脚本执行成功")
    except TimeoutError:
        p.kill()
        print("abaqus脚本运行失败，请检查代码")
        exit(-1)
    print(p.returncode)
    print("cae保存位置为：" + _config["cae_path"])
    '''

mould_name = sys.argv[1]
step = int(sys.argv[2])
if __name__ == "__main__":
    # 这里用某种方式得到一个列表
    param_list = []
    with open("/Optimizing_bending_parameter/data/mould_output/" + mould_name + "/param_base_rel.csv") as f:
        param_list = list(map(lambda x: list(map(float, x.split(","))), f.readlines()))

    gen_abaqus_model(
        data_path="/Optimizing_bending_parameter/data/model/" + mould_name,
        # param_list=
        user_config={
        # 矩形截面形状，对角线上的两个点
        "strip_section_shape": ((-0.2, -2.0), (0, 2.0)),
        "strip_length": 50,
        # Mesh 密度，长 高 宽的点的数量
        "mesh_num": (60, 2, 6),
        "param_list": param_list,
        "mass_scaling": 200,
    },
        step = step,
    )