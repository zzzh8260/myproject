import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import join, basename
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import sleep
from random import choice
import numpy as np
import pickle
import subprocess
# from utils import * # 移除了 utils.py 对自身的导入
from time import time
from datetime import datetime
from tensorflow.keras.losses import categorical_crossentropy
from functools import reduce
# 移除了 comet_ml 导入
# from comet_ml.query import Metric, Metadata, Parameter, Tag, Other
# import comet_ml
import sys
from mpi4py import MPI

mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))


# 移除了 comet_ml.api.API
# api = comet_ml.api.API()

# 移除了 read_comet_config 和 cometconfig

def count_available_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"])
        return str(output).count('UUID')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not found or failed; assuming 0 GPUs.")
        return 0


def _get_basename(name):
    name = '/'.join(name.split('/')[2:])
    return name.split(':')[0]


def _reshape_labels_like_logits(labels, logits, batchsize, nclass=10):
    return tf.reshape(tf.one_hot(labels, nclass), [batchsize, nclass])


def metrics(labels, logits, batchsize, reverse_ce=False):
    with tf.variable_scope('metrics'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        if not reverse_ce:
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_reshaped, logits=logits), name='xent')
        else:
            preds = tf.nn.softmax(logits)
            xent = tf.reduce_mean(categorical_crossentropy(labels_reshaped, 1 - preds))
        equal = tf.equal(labels, tf.cast(tf.argmax(logits, axis=1), dtype=labels.dtype))
        acc = tf.reduce_mean(tf.to_float(equal), name='acc')
    return xent, acc


def carlini(labels, logits, batchsize, clamp=-100):
    with tf.variable_scope('carlini'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        labels_reshaped = tf.cast(labels_reshaped, dtype=logits.dtype)
        target_logit = tf.reduce_sum(logits * labels_reshaped, axis=1)
        second_logit = tf.reduce_max(logits - logits * labels_reshaped, axis=1)
        cw_indiv = tf.maximum(second_logit - target_logit, clamp)
        # return tf.maximum(second_logit - target_logit, clamp)  # , target_logit, second_logit, tmp
        return tf.reduce_mean(cw_indiv)  # , target_logit, second_logit, tmp


def count_params_in_scope():
    scope = tf.get_default_graph().get_name_scope()
    nparam = sum([np.prod(w.shape.as_list()) for w in tf.trainable_variables(scope)])
    # print('scope:', scope, '#params', nparam)
    return nparam


def imagesc(img, title=None, fpath=None, experiment=None, step=None, scale='minmax'):
    if scale == 'minmax':
        img = img - img.ravel().min()
        img = img / img.ravel().max()
    elif type(scale) is float or type(scale) is int:  # good for perturbations
        img = img * .5 / scale + .5
    elif type(scale) is list or type(scale) is tuple:  # good for images
        assert len(scale) == 2, 'scale arg must be length 2'
        lo, hi = scale
        img = (img - lo) / (hi - lo)
    plt.clf()
    plt.imshow(img)
    if title:
        plt.title(title)

    # 移除了 experiment.log_figure
    # 现在改为保存到 fpath (如果提供)
    if fpath:
        try:
            plt.savefig(fpath)
        except Exception as e:
            print(f"Failed to save figure {fpath}: {e}")


def pgdstep(img, grad, orig, stepsize=.01, epsilon=.08, perturb=False):
    if perturb: img += (np.random.rand(*img.shape) - .5) * 2 * epsilon
    img += stepsize * np.sign(grad)
    img = np.clip(img, orig - epsilon, orig + epsilon)
    img = np.clip(img, 0, 255)
    return img


def l2_weights(weights):
    return tf.add_n([tf.reduce_sum(weight ** 2) for weight in weights.values() if len(weight.shape.as_list()) > 1])


def tf_preprocess(inputs, batchsize):
    print('data augmentation ON')
    # preprocessing data augmentation
    inputs = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]])
    inputs = tf.random_crop(inputs, [batchsize, 32, 32, 3])
    inputs = tf.map_fn(tf.image.random_flip_left_right, inputs)
    return inputs


def avg_n_dicts(dicts, experiment=None, step=None):
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means: means[key] = 0
            means[key] += dic[key] / len(dicts)
    # 移除了 experiment.log_metrics
    return means


def merge_n_dicts(dicts):
    # given a list of dicts with mutually exclusive schema, return a dict of all key-value pairs merged
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out


def plot_dict_series(dict_series, prefix=None, fpath_prefix=None, experiment=None, step=None):
    # given a list of dicts with the same schema, make a series plot for each key in the schema
    # if dict_series is a list of list of dicts, then overlap all plots in the second nested list
    serialized = {}
    for i, timestep in enumerate(dict_series):
        if type(timestep) is dict: timestep = [timestep]
        for dic in timestep:
            for key, val in dic.items():
                if key not in serialized: serialized[key] = []
                if len(serialized[key]) <= i: serialized[key].append([])
                serialized[key][-1].append(val)
    for key, series in serialized.items():
        plt.clf()
        plt.plot(np.array(series))
        plt.title('step {}'.format(step))
        plt.ylabel(key)

        # 移除了 experiment.log_figure
        if fpath_prefix:
            try:
                plt.savefig(f'{fpath_prefix}_{prefix}_{key}_step{step}.png')
            except Exception as e:
                print(f"Failed to save figure: {e}")


# 移除了 copy_to_args_from_experiment
# 移除了 comet_pull_weight
# 移除了 comet_pull_weight_by_key
# 移除了 comet_log_asset_weights_and_buffers
# 移除了 comet_log_asset

def transpose_list_of_lists(l):
    return list(map(list, zip(*l)))


def set_available_gpus(args):
    if len(args.gpu) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    else:
        args.gpu = list(range(count_available_gpus()))
    return args.gpu


class Dummy:
    def __getattribute__(self, attr):
        return lambda *arg, **kwarg: None


def lr_schedule(lrnrate, epoch, warmupperiod=5, schedule=[100, 150, 200], max_epoch=250):
    if schedule is None:
        schedule = [max_epoch // 2.667, max_epoch // 1.6, max_epoch // 1.142]
    warmupfactor = min(1, (epoch + 1) / (1e-6 + warmupperiod))
    if epoch < schedule[0]:
        return 1e00 * lrnrate * warmupfactor
    elif epoch < schedule[1]:
        return 1e-1 * lrnrate * warmupfactor
    elif epoch < schedule[2]:
        return 1e-2 * lrnrate * warmupfactor
    else:
        return 1e-3 * lrnrate * warmupfactor


def cr_schedule(craftrate, craftstep, warmupperiod=5, schedule=[20, 40]):
    warmupfactor = min(((craftstep + 1) / warmupperiod) ** 2, 1)
    if craftstep < schedule[0]:
        return 1e00 * craftrate * warmupfactor
    elif craftstep < schedule[1]:
        return 1e-1 * craftrate * warmupfactor
    else:
        return 1e-2 * craftrate * warmupfactor


def epochmass(epoch):
    return min(epoch / 5, 1)


def appendfeats(feats, feat, victimfeed, ybase, ytarget, batchsize):
    # feats is a defaultdict of type list which stores a 50000xNdim matrix of features for the entire dataset
    # feat is the minibatch of features to append
    cleaninputs, cleanlabels = [value for key, value in victimfeed.items() if 'adapter-0/cleaninputs' in str(key)][0]
    cleanmask = [value for key, value in victimfeed.items() if 'cleanmask' in str(key)][0]
    poisonmask = [value for key, value in victimfeed.items() if 'poisonmask' in str(key)][0]
    npoison = sum(poisonmask)
    feats['targetfeats'] = feat[batchsize:]
    feats['targetlabels'] = ytarget
    feats['cleanfeats'].extend(feat[npoison:batchsize])
    feats['poisonfeats'].extend(feat[:npoison])
    feats['cleanlabels'].extend(cleanlabels[cleanmask])
    feats['poisonlabels'].extend(ybase[poisonmask])


def get_featdist(feats):
    targetfeats, poisonfeats = feats['targetfeats'], feats['poisonfeats']
    targetfeat = np.array(targetfeats[:1])
    poisonfeats = np.array(poisonfeats)
    featdist = np.mean(np.linalg.norm(poisonfeats - targetfeat, axis=1))
    return featdist


# 移除了 uid2craftkey
# 移除了 get_param

def print_command_and_args(args):
    command = 'python ' + ' '.join(sys.argv)
    print(command)
    if rank == 0:
        print('\n'.join([f'{key} == {val}' for key, val in sorted(vars(args).items())]))
    return command


def assets2dict(assets, keystr, valuestr):
    # helps when doing comet api.get_something_summary() and it returns a list of dicts all with the attribute 'name'
    return {asset[keystr]: asset[valuestr] for asset in assets}


def tf_basename(tensor):
    name = tensor.name
    name = basename(name)
    if ':' in name:
        name = name.split(':')[0]
    return name


def trunc_decimal(val):
    if val > 1e10: return 'inf'
    return int(val * 100) / 100


def package_poisoned_dataset(poisoninputs, xtrain, ytrain, xtarget, ytarget, ytargetadv, xvalid, yvalid, args,
                             craftstep):
    start = int(args.poisonclass / (max(ytrain) + 1) * len(xtrain))
    xtrain[start: start + args.npoison] = poisoninputs
    asset = dict(xtrain=xtrain, ytrain=ytrain, xtarget=xtarget, ytarget=ytarget, ytargetadv=ytargetadv, xvalid=xvalid,
                 yvalid=yvalid)
    file = f'{args.poisondatasetfile}-{craftstep}.pkl'
    with open(file, 'wb') as f: pickle.dump(asset, f)
    print(
        f'argument -savepoisondataset is ON: poison dataset saved for expt {args.craftkey} craftstep {craftstep} at {file}')

# 移除了 comet_log_asset_apiexpt