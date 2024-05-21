import os
import os.path as osp
import sys
import datetime
# from mmcv import Config as MMConfig
from mmengine.config import Config as MMConfig


import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:

    def get_config_fromfile(self, config_path):
        self.config_path = config_path
        cfg = MMConfig.fromfile(self.config_path)
        self.__dict__.update(dict(cfg))

        # update dir
        self.cur_dir = osp.dirname(osp.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

        # add some paths to the system root dir
        sys.path.insert(0, osp.join(self.root_dir, 'common'))
        add_pypath(osp.join(self.data_dir))
        for dataset in os.listdir(osp.join(self.root_dir, 'data')):
            if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
                add_pypath(osp.join(self.root_dir, 'data', dataset))
        add_pypath(osp.join(self.root_dir, 'data'))
        add_pypath(self.data_dir)

    
                
    def prepare_dirs(self, exp_name = 'exp'):
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
        # self.output_dir = osp.join(self.root_dir, f'{exp_name}')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir, 'result')

        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        # os.makedirs(self.vis_dir, exist_ok=True)

        # ## copy some code to log dir as a backup
        # copy_files = ['main/train.py', 'main/test.py', 'common/base.py',
        #               'common/nets', 'main/SMPLer_X.py',
        #               'data/dataset.py', 'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py']
        # for file in copy_files:
        #     os.system(f'cp -r {self.root_dir}/{file} {self.code_dir}')

    def update_test_config(self, testset, agora_benchmark, shapy_eval_split, pretrained_model_path, use_cache,
                           eval_on_train=False, vis=False):
        self.testset = testset
        self.agora_benchmark = agora_benchmark
        self.pretrained_model_path = pretrained_model_path
        self.shapy_eval_split = shapy_eval_split
        self.use_cache = use_cache
        self.eval_on_train = eval_on_train
        self.vis = vis

    def update_config(self, num_gpus, exp_name):
        self.num_gpus = num_gpus
        self.exp_name = exp_name
        
        self.prepare_dirs(self.exp_name)
        
        # Save
        cfg_save = MMConfig(self.__dict__)
        cfg_save.dump(osp.join(self.code_dir,'config_base.py'))


# 确保配置在程序开始时初始化
cfg = Config()
