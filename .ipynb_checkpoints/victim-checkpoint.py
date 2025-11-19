print('loading modules victim')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
from parse import get_parser
from meta import Meta
from data import *
from utils import *
import pickle
import json
from time import time, sleep
from socket import gethostname
from collections import defaultdict
from mpi4py import MPI
import uuid

# initialize mpi
mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

# 移除 Comet API 相关
# api = API()
# weightapi = API()
# cometconfig = read_comet_config()

parser, attrs = get_parser(True)
for exclude in ['gpu']: attrs.remove(exclude)
args = parser.parse_args()

# --- Comet 逻辑移除 ---
# args.craftkey = args.uid if len(args.uid) == 32 else uid2craftkey(args.uid, args.craftproj)
# craftexpt = api.get_experiment(cometconfig["workspace"], args.craftproj, args.craftkey)
# copy_to_args_from_experiment(args, craftexpt, attrs)

# 本地逻辑：假设 craft uid (args.uid) 是我们加载产物的标识符
args.craftkey = args.uid
print(f"Running victim evaluation for craft uid: {args.craftkey}")

# 手动应用 X-args 覆盖
# if args.Xnvictimepoch > 0: args.nvictimepoch = args.Xnvictimepoch
# if args.Xntrial > 0: args.ntrial = args.Xntrial
# if args.Xvictimproj is not None: args.victimproj = args.Xvictimproj # 已移除
# if args.Xtag is not None: args.tag = args.Xtag # 已移除
if args.Xweightdecay: args.weightdecay = True
if args.Xaugment: args.augment = True
if args.Xbatchsize:
    args.batchsize *= 2
    args.nbatch /= 2
if args.Xlrnrate: args.lrnrate *= 2
if args.Xschedule: args.schedule = [200, 250, 300]
if args.Xnpoison is not None: args.npoison = args.Xnpoison
if args.Xnet is not None: args.net = args.Xnet
args.gpu = set_available_gpus(args)

global meta, sess, xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv

# 创建本地输出目录
LOCAL_OUTPUT_DIR = 'output'
ARTIFACT_DIR_BASE = os.path.join(LOCAL_OUTPUT_DIR, 'artifacts')
# 假设 craft uid (args.uid) 是用于加载的
ARTIFACT_DIR_LOAD = os.path.join(ARTIFACT_DIR_BASE, args.craftkey) 
VICTIM_LOG_DIR = os.path.join(LOCAL_OUTPUT_DIR, 'victim_logs')
if rank == 0:
    os.makedirs(VICTIM_LOG_DIR, exist_ok=True)

# 用于替换 Experiment 对象的桩类
class Dummy:
    def __getattribute__(self, attr):
        return lambda *arg, **kwarg: None

def victim(kwargs=None):

    def load_poison_local(craftstep):
        """从本地文件系统加载 poisons"""
        fname = f'poisoninputs-{craftstep}.pkl'
        fpath = os.path.join(ARTIFACT_DIR_LOAD, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: Poison file not found at {fpath}")
            return None
            
        for attempt in range(5):
            try:
                with open(fpath, 'rb') as f:
                    poisoninputs = pickle.load(f)
                if localrank == 0: print(f'==> poisoninputs-{craftstep} loaded from {fpath}')
                return poisoninputs[:args.npoison]
            except Exception as e:
                print(f'WARNING: Local pull attempt for craftstep {craftstep} failed on attempt {attempt}: {e}')
                sleep(5)
        return None

    if kwargs is not None:
        for key in kwargs: globals()[key] = kwargs[key]
        for key in argsmod: setattr(args, key, argsmod[key])
        
    print('==> begin victim train')
    trial = 0
    while args.ntrial is None or trial < args.ntrial:
        for craftstep in args.craftsteps:
            experiment = Dummy() # 替换 Experiment
            
            # 为本次 victim 运行设置本地日志
            victim_run_id = f"victim_{args.craftkey[:8]}_step{craftstep}_trial{trial}_{uuid.uuid4().hex[:8]}"
            metrics_log_path = os.path.join(VICTIM_LOG_DIR, f'metrics_{victim_run_id}.csv')
            config_log_path = os.path.join(VICTIM_LOG_DIR, f'config_{victim_run_id}.json')
            
            if localrank == 0:
                print(f"Logging victim run to {VICTIM_LOG_DIR}/{victim_run_id}")
                try:
                    log_params = vars(args).copy()
                    log_params['craftstep'] = craftstep
                    log_params['trial'] = trial
                    with open(config_log_path, 'w') as f:
                        json.dump(log_params, f, indent=4, default=lambda o: '<not serializable>')
                except Exception as e:
                    print(f"Failed to log victim config: {e}")

            if 'victim.py' in sys.argv[0]:
                poisoninputs = load_poison_local(craftstep) # 替换 comet_pull_poison
                if poisoninputs is None: 
                    print(f'skipping craftstep {craftstep}'); 
                    continue
                if args.savepoisondataset: 
                    package_poisoned_dataset(poisoninputs, xtrain, ytrain, xtarget, ytarget, ytargetadv, xvalid, yvalid, args, craftstep)
                    continue
                
                meta.global_initialize(args, sess)
                meta.poisoninputs.load(poisoninputs, sess)
                
            trainstep = 0
            for epoch in range(args.nvictimepoch):
                tic = time()
                lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod, args.schedule)

                # 移除了 logfeat (log hidden layer features)

                # log validation acc
                resV = {}
                if epoch in np.round((args.nvictimepoch - 1) * np.linspace(0, 1, args.nvalidpoints) ** 2):
                    resVs = []  # validation
                    for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
                        resV = sess.run(meta.resultV, validfeed)
                        resVs.append(resV)
                    resV_avg = avg_n_dicts(resVs)
                    
                    # 记录验证指标
                    if localrank == 0:
                        log_data_val = {'step_type': 'validation', 'trainstep': trainstep, 'epoch': epoch, **resV_avg}
                        log_metrics_local(metrics_log_path, log_data_val)

                # train one epoch
                resL = {}
                for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
                    _, resL = sess.run([meta.trainop, meta.resultL,], victimfeed)
                    
                    if not trainstep % 200 and localrank == 0: 
                        # 记录训练指标
                        log_data_train = {'step_type': 'train', 'trainstep': trainstep, 'epoch': epoch, **resL}
                        log_metrics_local(metrics_log_path, log_data_train)
                        
                    trainstep += 1
                    
                # 移除了 saveweights (comet_log_asset_weights_and_buffers)

                # *** 这是修正的地方 ***
                # 移除了对 {args.tag} 的引用
                # 确保这个 print 语句正确缩进在 for epoch 循环内部
                if not epoch % 20 and localrank == 0:
                    print(' | '.join([f'{args.craftkey[:5]} | trial-{trial} | craftstep-{craftstep} | epoch {epoch} | elapsed {round(time() - tic, 2)}'] +
                                     [f'{key} {trunc_decimal(val)}' for key, val in resL.items() if 'class' not in key] +
                                     [f'{key} {trunc_decimal(val)}' for key, val in resV.items() if 'class' not in key]))
            
            if localrank == 0:
                print(f"Victim run {victim_run_id} finished.")
            
        trial += 1

def log_metrics_local(log_path, metrics_dict):
    """辅助函数，用于将指标字典追加到CSV文件"""
    try:
        # 确保所有值都是可序列化的基本类型
        metrics_dict_serializable = {}
        for k, v in metrics_dict.items():
            try:
                # 尝试将 numpy/tf 类型转换为 python 基本类型
                metrics_dict_serializable[k] = v.item() 
            except AttributeError:
                # 已经是基本类型
                metrics_dict_serializable[k] = v
            except Exception:
                # 其他无法转换的类型
                metrics_dict_serializable[k] = str(v)

        # 获取所有可能的键，以保持列顺序一致
        header_keys = []
        if not os.path.exists(log_path):
            header_keys = metrics_dict_serializable.keys()
            with open(log_path, 'w') as f:
                f.write(','.join(header_keys) + '\n')
        else:
            # 读取现有 header
            with open(log_path, 'r') as f:
                header_keys = f.readline().strip().split(',')
            
            # 检查是否有新键
            new_keys = [k for k in metrics_dict_serializable.keys() if k not in header_keys]
            if new_keys:
                print(f"Warning: New metrics found {new_keys}, appending to header (CSV may become misaligned if keys are not consistent)")
                header_keys.extend(new_keys)
                # (为简单起见，我们不重写文件，只在追加时使用新 header)


        with open(log_path, 'a') as f:
            # 始终按照 header 顺序写入
            f.write(','.join([str(metrics_dict_serializable.get(k, '')) for k in header_keys]) + '\n')
            
    except Exception as e:
        # 避免日志记录失败导致程序崩溃
        print(f"Warning: Failed to log metrics to {log_path}. Error: {e}")


def landscape():
    # 移除了 Comet 相关的权重加载逻辑
    print("Landscape function was dependent on Comet.ml for weight loading and has been disabled.")
    pass


if __name__ == '__main__':
    ## load data and build graph
    print('==> loading data')
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv = load_and_apportion_data(mpi, args)
    print('==> building graph')
    meta = Meta(args, xbase, ybase, xtarget, ytarget, ytargetadv, victim=True)

    # start tf session and initialize variables
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    victim()
    # landscape()