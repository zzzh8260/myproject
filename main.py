print('loading modules')
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
from mpi4py import MPI
import warnings
import multiprocessing  # Just for threadcounting in rank0
from subprocess import Popen, STDOUT, PIPE
import socket
import uuid

# initialize mpi
mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = get_parser()
args = parser.parse_args()
args.gpu = set_available_gpus(args)
ncpu = multiprocessing.cpu_count()
print('==> Rank {}/{}, localrank {}, host {}, GPU {}/{}, nCPUs {}'.format(rank, nproc, localrank, socket.gethostname(),
                                                                          localrank % len(args.gpu), len(args.gpu),
                                                                          ncpu))
args.nproc = nproc
args.nmeta = args.nproc * args.nreplay
args.maxepoch = args.nmeta * args.stagger

# 创建本地输出目录
LOCAL_OUTPUT_DIR = 'output'
ARTIFACT_DIR = os.path.join(LOCAL_OUTPUT_DIR, 'artifacts', args.uid)
CHECKPOINT_DIR = os.path.join(LOCAL_OUTPUT_DIR, 'checkpoints', args.uid)
FIGURE_DIR = os.path.join(LOCAL_OUTPUT_DIR, 'figures', args.uid)
if rank == 0:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


# 用于替换 Experiment 对象的桩类
class Dummy:
    def __getattribute__(self, attr):
        return lambda *arg, **kwarg: None


def craft():
    experiment = Dummy()  # 替换 Experiment 对象

    # 在 rank 0 上保存配置
    if rank == 0:
        config_path = os.path.join(LOCAL_OUTPUT_DIR, f'config_{args.uid}.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=4)
            print(f'Logged config to {config_path}')
        except Exception as e:
            print(f'Failed to log config: {e}')
        print(f'Running with uid: {args.uid}')

    def save_poisons_local(craftstep):
        """将 poisons 和相关图像保存到本地文件"""
        poisoninputs = sess.run(meta.poisoninputs, {meta.poisonmask: [True] * args.npoison})

        # 保存 poison 产物
        fname = f'poisoninputs-{craftstep}.pkl'
        fpath = os.path.join(ARTIFACT_DIR, fname)
        try:
            with open(fpath, 'wb') as f:
                pickle.dump(poisoninputs, f)
        except Exception as e:
            print(f'Failed to save artifact {fname}: {e}')

        # 保存图像
        npoison_to_display = 10
        for i in np.linspace(0, args.npoison - 1, npoison_to_display, dtype=int):
            fpath_fig = os.path.join(FIGURE_DIR, f'poison-{i}-step{craftstep}.png')
            imagesc(poisoninputs[i], title=f'poison-{i}', fpath=fpath_fig, scale=[0, 255])

            fpath_fig_pert = os.path.join(FIGURE_DIR, f'perturb-{i}-step{craftstep}.png')
            imagesc(poisoninputs[i] - xbase[i], title=f'perturb-{i}', fpath=fpath_fig_pert, scale=127.5)

        for i in range(len(xtarget)):
            fpath_fig_targ = os.path.join(FIGURE_DIR, f'target-{i}-step{craftstep}.png')
            imagesc(xtarget[i], title=f'target-{i}', fpath=fpath_fig_targ, scale=[0, 255])

    def restart_poison():
        perts = np.random.uniform(-args.eps, args.eps, xbase.shape)
        perts = np.clip(xbase + perts, 0, 255) - xbase
        mpi.Bcast(perts, root=0)
        meta.restart_poison(perts, sess)

    def log_epoch_results(resMs, resLs, craftstep):
        resMgather = mpi.gather(resMs, root=0)
        resLgather = mpi.gather(resLs, root=0)
        if rank == 0:
            resMgather = sum(resMgather, [])  # flattens the list of lists
            resLgather = sum(resLgather, [])
            resM, resL = avg_n_dicts(resMgather), avg_n_dicts(resLgather)

            # 打印到控制台
            print(' | '.join(['craftstep {}'.format(craftstep)] + ['elapsed {}'.format(round(time() - tic, 3))] +
                             ['{} {}'.format(key, round(val, 2)) for key, val in resM.items()]))

            # (可选) 写入本地 metrics.csv
            try:
                metrics_log_path = os.path.join(LOCAL_OUTPUT_DIR, f'metrics_craft_{args.uid}.csv')
                log_data = {'craftstep': craftstep, **resM, **resL}

                # 简化版CSV日志记录
                if not os.path.exists(metrics_log_path):
                    with open(metrics_log_path, 'w') as f:
                        f.write(','.join(log_data.keys()) + '\n')
                with open(metrics_log_path, 'a') as f:
                    f.write(','.join([str(log_data[k]) for k in log_data.keys()]) + '\n')
            except Exception as e:
                print(f"Failed to log metrics to file: {e}")

    def dock_weights_and_buffers(epoch, craftstep):
        if epoch == 0:  # randomly initialize
            meta.init_weights(sess, pretrain_weights)
        elif craftstep == 0:  # train or load from weightset to correct epoch
            if args.weightsettrain or args.weightset == '':
                train(epoch)
            else:
                # 替换 comet_pull_weight
                # meta.load_weights(sess, comet_pull_weight(epoch, weightapi, args, rank))
                print(f"Rank {rank}: Skipping weight loading, implement local loading if needed.")
                pass  # 需要实现本地权重加载逻辑
        else:  # restore weights from previous replay
            meta.cache_weights(sess, cache=f'replay-{replay}', restore=True)

    print('==> begin crafting poisons on rank {}'.format(rank))
    for craftstep in range(args.ncraftstep):
        # auxiliary tasks
        tic = time()
        if not craftstep % args.restartperiod: restart_poison()
        if not craftstep % args.logperiod and rank == 0: save_poisons_local(craftstep)  # 替换 comet_log_poison
        craftrate = cr_schedule(args.craftrate, craftstep, schedule=[i * args.crdropperiod for i in [1, 2]])

        resMs, resLs = [], []
        for replay in range(args.nreplay):
            epoch = ((rank + replay * args.nproc) * args.stagger + craftstep) % (args.nmeta * args.stagger)
            lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
            dock_weights_and_buffers(epoch, craftstep)

            # iterate through all batches in epoch
            for craftfeed, trainfeed, hasPoison in feeddict_generator(xtrain, ytrain, lrnrate, meta, args):
                if args.trajectory == 'clean':
                    if hasPoison: _, resM, = sess.run([meta.accumop, meta.resultM, ], craftfeed)
                    _, resL, = sess.run([meta.trainop, meta.resultL, ], trainfeed)
                elif args.trajectory == 'poison':
                    _, _, resM, resL, = sess.run([meta.accumop, meta.trainop, meta.resultM, meta.resultL, ], craftfeed)
            meta.cache_weights(sess, cache=f'replay-{replay}')
            resL.update(dict(epoch=epoch, craftrate=craftrate))
            resMs.append(resM);
            resLs.append(resL)

        avg_metagrads = []
        for metagrad_accum in sess.run(meta.metagrads_accum):
            avg_metagrad = np.zeros_like(metagrad_accum)
            mpi.Allreduce(metagrad_accum, avg_metagrad, op=MPI.SUM)
            avg_metagrads.append(avg_metagrad / args.nmeta)
        sess.run([meta.craftop, ], {meta.avg_metagrads: tuple(avg_metagrads), meta.craftrate: craftrate})
        log_epoch_results(resMs, resLs, craftstep)

    print(f"Crafting finished for uid {args.uid}")

    if not args.skipvictim:
        print('==> crafting finished. begin victim.')
        meta.init_weights(sess, pretrain_weights)
        from victim import victim
        argsmod = dict(craftsteps=[craftstep], ntrial=1, Xtag=None)
        kwargs = dict(argsmod=argsmod, sess=sess, meta=meta, xtrain=xtrain, ytrain=ytrain, xvalid=xvalid, yvalid=yvalid,
                      xbase=xbase, ybase=ybase, xtarget=xtarget, ytarget=ytarget, ytargetadv=ytargetadv)
        victim(kwargs)


def train(nepoch):
    tic = time()
    expt = Dummy()  # 替换 Experiment

    # train and valid
    meta.init_weights(sess, pretrain_weights)  # reinitialize weights and buffers
    print(f'==> begin vanilla train on rank {rank} to epoch {nepoch}')
    for epoch in range(nepoch):
        lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
        for _, trainfeed, _ in feeddict_generator(xtrain, ytrain, lrnrate, meta, args):
            _, resL, = sess.run([meta.trainop, meta.resultL, ], trainfeed)
        # 移除了 expt.log_metrics

    resVs = []  # begin validation
    for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
        resV, = sess.run([meta.resultV, ], validfeed)
        resVs.append(resV)
    # 移除了 expt.log_metrics

    # log weights and buffers of final trained model to local file
    save_weights_local(nepoch, meta, sess)  # 替换 comet_log_asset_weights_and_buffers
    print(' | '.join(['trained to {}'.format(nepoch)] +
                     ['total time {}'.format(round(time() - tic, 3))] +
                     ['{} {}'.format(key, int(val * 100) / 100.) for key, val in resL.items() if 'class' not in key]))


def save_weights_local(epoch, meta_obj, session):
    """本地保存权重和缓冲区"""
    if rank != 0:
        return

    fpath = os.path.join(CHECKPOINT_DIR, f'weights0-{epoch}.pkl')
    try:
        with open(fpath, 'wb') as f:
            pickle.dump(session.run((meta_obj.weights0, meta_obj.buffers0)), f)
        print(f"Saved weights for epoch {epoch} to {fpath}")
    except Exception as e:
        print(f"Failed to save weights: {e}")


if __name__ == '__main__':
    # load data and build graph
    print('==> loading data on rank {}'.format(rank))
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv = load_and_apportion_data(mpi, args)
    print('==> building graph on rank {}'.format(rank))
    meta = Meta(args, xbase, ybase, xtarget, ytarget, ytargetadv)

    # start tf session and initialize variables
    print('==> initializing tf session on rank {}'.format(rank))
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    if args.justtrain == 0:
        craft()
    else:
        train(args.justtrain)