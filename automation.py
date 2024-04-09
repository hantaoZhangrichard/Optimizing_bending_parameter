import subprocess
import os
import shutil



def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    p.wait(2400)
    if p.poll() == 0:
        print(cmd[1], 'Success')
    else:
        print(cmd[1], 'Failure')

if __name__ == "__main__":
    dir1 = ".\data\model\\"
    dir2 = ".\data\mould_output\\"
    for i in range(0,1):
        mould_name = "test" + str(i)
        mould_name = "test00"
        print(mould_name)
        if not os.path.exists(dir1 + mould_name):
            os.makedirs(dir1 + mould_name)
        if not os.path.exists(dir2 + mould_name):
            os.makedirs(dir2 + mould_name)

        # tasks = ['gen_curve_and_mould.py', 'calc_init_param.py', 'gen_abaqus_model.py', 'gen_spring_back_model.py']

        # Get curve and mould

        cmd = ['python ', 'gen_curve_and_mould.py', mould_name]
        run_cmd(cmd)
        
        shutil.copy(dir2 + mould_name + '\\mould.stp', dir1 + mould_name)

        '''
        cmd = ['python ', tasks[1], mould_name]
        run_cmd(cmd)
        cmd = ['python ', tasks[2], mould_name]
        run_cmd(cmd)
        cmd = ['python ', tasks[3], mould_name]
        run_cmd(cmd)
        '''