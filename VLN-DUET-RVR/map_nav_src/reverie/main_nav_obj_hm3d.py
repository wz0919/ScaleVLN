import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

import sys
sys.path.append(".")
sys.path.append("../.")
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from utils.data import ImageFeaturesDB, ImageFeaturesDB2

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args

from reverie.env_hm3d import HM3DReverieObjectNavBatch

hm3d_dir = '../datasets/HM3D/'

def build_dataset(args, rank=0):
    tok = get_tokenizer(args)

    if args.aug is not None:
      aug_feat_db = ImageFeaturesDB(args.aug_ft_file, args.image_feat_size)
    val_feat_db = ImageFeaturesDB(args.val_ft_file, args.image_feat_size)
    train_feat_db = ImageFeaturesDB2(args.mp3d_ft_files, args.image_feat_size)

    # feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    dataset_class = ReverieObjectNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if not args.test and args.aug is not None:
        aug_instr_file = args.aug

        if args.features == "clip-h14":
            feat_db_file = "../datasets/R2R/features/clip_vit-h14_final.hdf5"
        else:
            feat_db_file = "../../generate_fts/img_fts"

        if args.obj_features == "timm_vitb16":
            obj_db_file = "../datasets/REVERIE/features/obj_fts"

        aug_env = HM3DReverieObjectNavBatch(
            feat_db_file, obj_db_file, [aug_instr_file], args.connectivity_dir, 
            batch_size=args.batch_size, max_objects=args.max_objects,
            image_feat_size=args.image_feat_size, obj_feat_size=args.obj_feat_size,
            angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            multi_endpoints=False, multi_startpoints=False,
            scan_ranges=args.hm3d_scan_ranges,
        )
    else:
        aug_env = None

    if args.aug_only:
        train_env, aug_env = aug_env, None
        args.aug = None
    else:
        train_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['train'], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        train_env = dataset_class(
            train_feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
            num_mp3d_scans=args.num_mp3d_scans
        )

    # val_env_names = ['val_train_seen']
    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}

    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            val_feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )

    # exit()
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
        # return

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback, og_loss_weight=args.hm3d_og_loss_weight)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )

def zero_shot(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)
    agent_class = GMapObjectNavAgent

    if os.path.exists(os.path.join(args.log_dir, 'zero_shot_eval.json')):
        zero_shot_record = json.load(open(os.path.join(args.log_dir, 'zero_shot_eval.json')))
    else:
        zero_shot_record = {}

    evaludated = list(zero_shot_record.keys())

    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state":""}}
    else:
        best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
        for k,v in zero_shot_record.items():
            if v['val_unseen']['spl'] + v['val_unseen']['sr'] > best_val['val_unseen']['spl'] + best_val['val_unseen']['sr']:
                best_val['val_unseen']['spl'] = v['val_unseen']['spl']
                best_val['val_unseen']['sr'] = v['val_unseen']['sr']
                best_val['val_unseen']['state'] = v['val_unseen']['state']


    if default_gpu:
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'zero_shot.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    import glob
    models_paths = list(
        filter(os.path.isfile, glob.glob(args.ckpt_dir))
    )
    models_paths.sort(key=os.path.getmtime)

    # start = time.time()
    while True:
        current_ckpt = None
        while current_ckpt is None:
            checkpoint_folder = models_paths
            if False: # future ceph
                models_paths = [p for p in filter(os.path.isfile, glob.glob(checkpoint_folder + "/*")) if p not in evaluated]
            else:
                models_paths = [p for p in list(
                    filter(os.path.isfile, glob.glob(args.ckpt_dir+"/*"))
                ) if p not in evaludated]
                models_paths.sort(key=os.path.getmtime)
            if len(models_paths) > 0:
                models_paths.sort(key=os.path.getmtime)
                current_ckpt = models_paths[0]
                idx = models_paths[0].split('/')[-1].split('.')[0].split('_')[-1]
                iter = idx
            else:
                current_ckpt = None
            time.sleep(20)  # sleep for 2 secs before polling again

        zero_shot_record[current_ckpt] = {}
        args.bert_ckpt_file = current_ckpt

        listner = agent_class(args, train_env, rank=rank)
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}".format(current_ckpt),
                record_file
            )

        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                zero_shot_record[current_ckpt][env_name] = score_summary
                zero_shot_record[current_ckpt][env_name]['state'] = 'Iter %s' % (loss_str)

                # select model by spl+sr
                if env_name in best_val:
                    if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %s' % (loss_str)

        write_to_record_file(
            loss_str,
            record_file
        )
        write_to_record_file("BEST RESULT TILL NOW", record_file)
        for env_name in best_val:
            write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)

        with open(os.path.join(args.log_dir, 'zero_shot_eval.json'), 'w') as outf:
            json.dump(zero_shot_record, outf, indent=4)
        evaludated.append(current_ckpt)       

def main():
    print('begin parsing')
    args = parse_args()
    print('finish parsing')

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    print('building datasets')
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        if args.zero_shot:
            zero_shot(args, train_env, val_envs, aug_env=aug_env, rank=rank)
        else:
            train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
