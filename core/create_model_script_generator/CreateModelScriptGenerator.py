# encoding:utf-8
"""
    脚本生成器
"""
import sys

from ._default_scripts import *


class CreateModelScriptGenerator:
    def __init__(self):
        self.config = {
            # 基本的输出文件及输出文件模式
            "output_file": None,
            "io_mode": "w",
            # 模型的名称
            "model_name": None,
            "new_model": False,
            # 模具方块的边长的一半
            "clamp_half_length": 26,
            # 铝件信息
            "strip_length": None,
            "strip_section_file": None,
            # 生成模具用
            "mould_file": None,
            # 材料参数，一个字典
            "material": {},
            # 生成job
            "cpu_num": 32,
            # mesh相关
            "gen_mesh": None,
            "mesh_file": None,
            # step相关
            "mass_scaling": None,
            "param_list": None,
            "step_config": {
                "has_pre_length": True,  # 暂时可能没啥用
                "global_time_period": None,
                "specified_time_period": {},  # 为某一步单独设置时间
                "basic_amp_table": None,  # 默认的amp曲线
                "amp_type": None,
                "only_use_amp_table": None,
                "start_idx_while_using_amp_table": None,
            },
            # 保存的CAE的位置
            "cae_path": None,
        }

        self._script_io_wrapper = None

    # noinspection DuplicatedCode
    def set_config(self, _config: dict):
        self.config.update(_config)

    def _check_config(self, *args):
        wrong_cnt = 0
        for param in args:
            if param not in self.config or self.config[param] is None:
                sys.stderr.write(f"You must set {param}")
                wrong_cnt += 1
        if wrong_cnt > 0:
            exit(-1)

    def  gen_all(self):
        self._script_io_wrapper = open(self.config["output_file"], "w", encoding="utf-8")
        self._gen_header_script()
        self._gen_clamp_script()
        self._gen_mould_script()
        self._gen_strip_scrip()
        self._gen_material_script()
        self._gen_mesh()
        self._gen_assembly_script()
        # self._gen_step_script()
        self._gen_job_script()
        self._gen_save_model()
        self._script_io_wrapper.close()
        self._script_io_wrapper = None

    def _gen_header_script(self):
        self._check_config("new_model", "model_name")
        self._script_io_wrapper.write(HEADER_SCRIPT_STR)
        self._script_io_wrapper.write("\nfrom abaqus import *\nfrom abaqusConstants import *\nfrom caeModules import *\nfrom driverUtils import executeOnCaeStartup\n")
        if self.config["new_model"]:
            self._script_io_wrapper.write(MODEL_SCRIPT_STR.format(model_name=self.config["model_name"]))

    def _gen_clamp_script(self):
        self._check_config("model_name", "clamp_half_length")
        self._script_io_wrapper.write(CLAMP_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            half_length=self.config["clamp_half_length"],
        ))

    def _gen_strip_scrip(self):
        self._check_config("model_name", "strip_section_file", "strip_length")
        self._script_io_wrapper.write(STRIP_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            strip_section_file=self.config["strip_section_file"],
            strip_length=self.config["strip_length"],
        ))

    def _gen_mould_script(self):
        self._check_config("model_name", "mould_file")
        self._script_io_wrapper.write(MOULD_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            mould_file=self.config["mould_file"],
        ))

    def _gen_material_script(self):
        self._check_config("model_name", "material")
        self._script_io_wrapper.write(MATERIAL_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            density=self.config["material"]["density"],
            young=self.config["material"]["young"],
            possion=self.config["material"]["possion"],
            plastic_table=self.config["material"]["plastic_str"],
        ))
        if self.config["material"]["has_damping"]:
            self._script_io_wrapper.write(DAMPING_SCRIPT_STR.format(
                model_name=self.config["model_name"],
                alpha=self.config["material"]["alpha"],
            ))

    def _gen_mesh(self):
        if not self.config["gen_mesh"]:
            return
        with open(self.config["mesh_file"], "r", encoding="utf-8") as mesh_f:
            content = mesh_f.read()
            self._script_io_wrapper.write(content)

    # noinspection PyPep8Naming
    def _gen_assembly_script(self):
        self._check_config("model_name")
        # 摩擦系数
        friction_coeff = 0.1
        Dyz = 76
        Dxy = 205
        D = 40
        X0 = self.config["strip_length"]
        Y0 = 0
        Z0 = 0
        self._script_io_wrapper.write(ASSEMBLY_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            X0=X0,
            Y0=Y0,
            Z0=Z0,
            Dyz_Dxy_D=Dyz + Dxy + D,
            Dyz_Dxy=Dyz + Dxy,
            Dxy=Dxy,
            friction_coeff=friction_coeff,
        ))

    def _gen_job_script(self):
        self._check_config("model_name", "cpu_num")
        self._script_io_wrapper.write(JOB_SCRIPT_STR.format(
            model_name=self.config["model_name"],
            cpu_num=self.config["cpu_num"],
        ))

    # def __add_first_step_str(self, step_name, previous_step_name, time_period):
    #     return FIRST_STEP_SCRIPT_STR.format(
    #         self.config["model_name"], self.config["mass_scaling"], step_name, previous_step_name, time_period
    #     )
    #
    # def __add_step_str(self, step_name, previous_step_name, time_period):
    #     return STEP_SCRIPT_STR.format(
    #         self.config["model_name"], self.config["mass_scaling"], step_name, previous_step_name, time_period
    #     )

    # noinspection PyPep8Naming
    def __add_BC_str(
            self,
            *,
            bc_name,
            amp,
            step_name,
            u1=None,
            u2=None,
            u3=None,
            ur1=None,
            ur2=None,
            ur3=None
    ):
        params = ""
        if u1 is not None:
            params += "u1=" + str(u1)
        if u2 is not None:
            if params != "":
                params += ", "
            params += "u2=" + str(u2)
        if u3 is not None:
            if params != "":
                params += ", "
            params += "u3=" + str(u3)
        if ur1 is not None:
            if params != "":
                params += ", "
            params += "ur1=" + str(ur1)
        if ur2 is not None:
            if params != "":
                params += ", "
            params += "ur2=" + str(ur2)
        if ur3 is not None:
            if params != "":
                params += ", "
            params += "ur3=" + str(ur3)
        return ADD_BC_SCRIPT_STR.format(self.config["model_name"], bc_name, amp, step_name, params)

    def _gen_step_script(self):
        # 添加基础幅值曲线
        self._script_io_wrapper.write(
            AMP_DICT[self.config["step_config"]["amp_type"]].format(
                model_name=self.config["model_name"],
                amp_table=str(tuple(self.config["step_config"]["basic_amp_table"])),
                amp_name="amp_basic",
            )
        )
        # 添加预拉伸
        self._script_io_wrapper.write(
            FIRST_STEP_SCRIPT_STR.format(
                model_name=self.config["model_name"],
                mass_scaling=self.config["mass_scaling"],
                step_name="Step-0",
                previous_step_name="Initial",
                time_period=self.config["step_config"]["global_time_period"],
            ))
        # 平移边界条件
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="translate_fix_rotate",
                amp="amp_basic",
                step_name="Step-0",
                ur1=0,
                ur2=0,
                ur3=0,
            ))
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="translate_x",
                amp="amp_basic",
                step_name="Step-0",
                u1=self.config["param_list"][0][0],  # 默认预拉伸是参数列表中第一个
            ))
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="translate_y",
                amp="amp_basic",
                step_name="Step-0",
                u2=0,
            ))
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="translate_z",
                amp="amp_basic",
                step_name="Step-0",
                u3=0,
            ))
        # z轴转动
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="rotate_z",
                amp="amp_basic",
                step_name="Step-0",
                ur1=0,
            ))
        # y轴转动
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="rotate_y",
                amp="amp_basic",
                step_name="Step-0",
                ur1=0,
            ))
        # x轴转动
        self._script_io_wrapper.write(
            self.__add_BC_str(
                bc_name="rotate_x",
                amp="amp_basic",
                step_name="Step-0",
                ur1=0,
            ))
        if not self.config["step_config"]["only_use_amp_table"]:
            self._gen_v1()
        else:
            self.__gen_v2()

    def _gen_v1(self, step=1):
        for index, val in enumerate(self.config["param_list"][step:step+1]):
            # 跳过预拉伸
            time_period = self.config["step_config"]["global_time_period"]
            # 创建分析步
            self._script_io_wrapper.write(STEP_SCRIPT_STR.format(
                model_name=self.config["model_name"],
                step_name=f"Step-{step+index}",
                previous_step_name=f"Step-{step+index-1}",
                time_period=time_period,
            ))
            # 添加边界条件
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="translate_x",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                u1=val[0],
            ))
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="translate_y",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                u2=val[1],
            ))
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="translate_z",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                u3=val[2],
            ))
            # z轴转动
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="rotate_z",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                ur1=val[5],
            ))
            # y轴转动
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="rotate_y",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                ur1=val[4],
            ))
            # x轴转动
            self._script_io_wrapper.write(self.__add_BC_str(
                bc_name="rotate_x",
                amp="amp_basic",
                step_name=f"Step-{step+index}",
                ur1=val[3],
            ))

    def __gen_v2(self):
        """
            使用单独的一张表
        """

        step_num = len(self.config["param_list"])
        time_period = self.config["step_config"]["global_time_period"]
        all_time_period = time_period * step_num
        amp_table_list = [[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)],
                          [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]]

        tab_param_list = self.config["param_list"][1:]
        # 计算幅值曲线表
        for index, val in enumerate(tab_param_list):
            for i in range(6):
                amp_table_list[i].append(
                    (time_period * (index + 1),
                     amp_table_list[i][index][1] + val[i])
                )
        amp_type = self.config["step_config"]["amp_type"]
        model_name = self.config["model_name"]
        # 生成幅值曲线
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[0])),
            amp_name="amp_translate_x"
        ))
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[1])),
            amp_name="amp_translate_y"
        ))
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[2])),
            amp_name="amp_translate_z"
        ))
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[3])),
            amp_name="amp_rotate_x"
        ))
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[4])),
            amp_name="amp_rotate_y"
        ))
        self._script_io_wrapper.write(AMP_DICT[amp_type].format(
            model_name=model_name,
            amp_table=str(tuple(amp_table_list[5])),
            amp_name="amp_rotate_z"
        ))
        # 创建分析步
        self._script_io_wrapper.write(STEP_SCRIPT_STR.format(
            model_name=model_name,
            step_name="Step-1",
            previous_step_name="Step-0",
            time_period=all_time_period,
        ))
        # 添加边界条件
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="translate_x",
            amp="amp_translate_x",
            step_name="Step-1",
            u1=1,
        ))
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="translate_y",
            amp="amp_translate_y",
            step_name="Step-1",
            u2=1,
        ))
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="translate_z",
            amp="amp_translate_z",
            step_name="Step-1",
            u3=1,
        ))
        # z轴转动
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="rotate_z",
            amp="amp_rotate_z",
            step_name="Step-1",
            ur1=1,
        ))
        # y轴转动
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="rotate_y",
            amp="amp_rotate_y",
            step_name="Step-1",
            ur1=1,
        ))
        # x轴转动
        self._script_io_wrapper.write(self.__add_BC_str(
            bc_name="rotate_x",
            amp="amp_rotate_x",
            step_name="Step-1",
            ur1=1,
        ))

    def _gen_save_model(self):
        self._script_io_wrapper.write("del mdb.models['Model-1']\n")
        self._script_io_wrapper.write("mdb.saveAs(pathName=\"{}\")\n\n".format(
            self.config["cae_path"]
        ))
    
    # Submit job
    def _gen_submit_job(self, job_name="Job-Model_base_0"):
        self._script_io_wrapper.write("executeOnCaeStartup()\n")
        self._script_io_wrapper.write("openMdb(\"{}\")\n".format(self.config["cae_path"]))
        self._script_io_wrapper.write("mdb.jobs['{}'].submit(consistencyChecking=OFF)\n".format(job_name))
        self._script_io_wrapper.write("mdb.save()\n\n")
