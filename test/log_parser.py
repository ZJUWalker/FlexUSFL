from datetime import datetime
import os
import math
import json

global log_dir


def get_all_log_files(dir: str):
    files = os.listdir(dir)
    log_files = []
    for file in files:
        if file.endswith('.log'):
            log_files.append(os.path.join(dir, file))
        elif os.path.isdir(os.path.join(dir, file)):
            log_files.extend(get_all_log_files(os.path.join(dir, file)))
    return log_files


def _get_epoch_time(fp: str):
    time_dict = {}
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'start with args' in line:
                start = line.index('start with args: ') + len('start with args: ')
                dict_data = line[start:].strip()
                dataset = eval(dict_data)['dataset']
                if dataset not in time_dict:
                    time_dict[dataset] = None
            if 'Finished' in line:
                start = line.index('train epoch time: ')
                end = start + len('train epoch time: ') + 6
                time_dict[dataset] = line[start + len("train epoch time: ") : end]
                # if dataset == 'gsm8k' and 'client_number_32' in fp:
                # print(f'fp: {fp}, time: {time_dict[dataset]}')
    return time_dict

def _get_epoch_time_ablation(fp: str):
    time_tuple = [None]*2
    cur=0
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Finished' in line:
                start = line.index('train epoch time: ')
                end = start + len('train epoch time: ') + 6
                time_tuple[cur] = line[start + len("train epoch time: ") : end]
                cur+=1
                # if dataset == 'gsm8k' and 'client_number_32' in fp:
                # print(f'fp: {fp}, time: {time_dict[dataset]}')
    return time_tuple


def _get_mem_alloc(fp: str):
    max_mem = 0
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if 'step' in line:
            values = line.split('|')
            # print(values[1].strip())
            try:
                max_mem = max(max_mem, float(values[1].strip()))
            except Exception as e:
                continue
    return max_mem

def _get_mem_alloc_ablation(fp: str):
    max_mem = [-1]*2
    cur=-1
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'step' in line:
                cur+=1
                continue
            values = line.split('|')
            # print(values[1].strip())
            try:
                max_mem[cur] = max(max_mem[cur], float(values[1].strip()))
            except Exception as e:
                continue
    return max_mem


def _get_server_loss_convergence(fp: str):
    losses = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split('|')
            try:
                losses.append(float(values[-1].strip()))
            except Exception as e:
                continue
    return losses

def _get_server_time(fp: str):
    times = []
    base=-1
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'step' not in line:
                time_str=line[:len('2025-07-28 12:28:33,098')]
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                # 计算从当天 00:00:00 起的秒数
                seconds_since_midnight = (
                    dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
                ).total_seconds()
                if base == -1:
                    base = seconds_since_midnight
                times.append(int(seconds_since_midnight-base))
    return times

def _get_client_loss_convergence(fp: str):
    losses = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'batch' in line and 'train epoch' in line:
                start = line.index('loss: ')
                end = start + len('loss: ') + 5
                l = line[start + len("loss: ") : end]
                losses.append(float(l))
    return losses

def get_avg_epoch_time_ablation(model, version):
    files = get_all_log_files(log_dir)
    epoch_time_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        avg_time = [0,0]
        target_files = list(
            filter(lambda x: model in x and version in x and 'server' not in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        # for f in target_files:
        #     print(f)
        for file in target_files:
            epoch_time = _get_epoch_time_ablation(file)
            # print(f'file: {file}, epoch_time: {epoch_time}')
            avg_time[0] += float(epoch_time[0]) if epoch_time[0] is not None else 0
            avg_time[1] += float(epoch_time[1]) if epoch_time[1] is not None else 0
            # avg_time += float(epoch_time)
        if len(target_files) > 0:
            avg_time[0] /= client_num/5
            avg_time[1] /= client_num/5
            epoch_time_dict[client_num] = avg_time
        else:
            epoch_time_dict[client_num] = None
    return epoch_time_dict  

def get_avg_epoch_time(model, version):
    files = get_all_log_files(log_dir)
    epoch_time_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        avg_time = {}
        target_files = list(
            filter(lambda x: model in x and version in x and 'server' not in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        # for f in target_files:
        #     print(f)
        for file in target_files:
            epoch_time = _get_epoch_time(file)
            # print(f'file: {file}, epoch_time: {epoch_time}')
            if epoch_time != {}:
                for k, v in epoch_time.items():
                    if k not in avg_time:
                        avg_time[k] = 0
                    avg_time[k] += float(v) if v is not None else 0
                # avg_time += float(epoch_time)
        if len(target_files) > 0:
            for k, v in avg_time.items():
                avg_time[k] /= client_num
            epoch_time_dict[client_num] = avg_time
        else:
            epoch_time_dict[client_num] = None
    return epoch_time_dict


def get_max_mem_alloc(model, version):
    files = get_all_log_files(log_dir)
    max_mem_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        max_mem = 0
        target_file = list(
            filter(lambda x: model in x and version in x and 'training_metrics' in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        if len(target_file) > 0:
            max_mem = _get_mem_alloc(target_file[0])
            max_mem_dict[client_num] = f'{max_mem:.4f}'
        else:
            max_mem_dict[client_num] = None
    return max_mem_dict


def get_max_mem_alloc_ablation(model, version):
    files = get_all_log_files(log_dir)
    max_mem_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        max_mem = 0
        target_file = list(
            filter(lambda x: model in x and version in x and 'training_metrics' in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        if len(target_file) > 0:
            max_mem = _get_mem_alloc_ablation(target_file[0])
            max_mem_dict[client_num] = max_mem
        else:
            max_mem_dict[client_num] = None
    return max_mem_dict


def get_client_loss_convergence(model, version):
    def average_n_elements(arr, n=5):
        if n == 1:
            return arr
        result = []
        for i in range(0, len(arr), n):
            # 取当前n个数的子数组
            sub_arr = arr[i : i + n]
            # 计算子数组的平均值
            avg = sum(sub_arr) / len(sub_arr)
            result.append(avg)
        return result

    files = get_all_log_files(log_dir)
    client_loss_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        avg_losses = []
        target_files = list(
            filter(lambda x: model in x and version in x and 'server' not in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        # for f in target_files:
        #     print(f)
        for file in target_files:
            losses = _get_client_loss_convergence(file)
            if losses != []:
                if avg_losses == []:
                    avg_losses = losses
                else:
                    try:
                        for i in range(len(avg_losses)):
                            avg_losses[i] += losses[i]
                    except Exception as e:
                        continue
        if len(target_files) > 0:
            for i in range(len(avg_losses)):
                avg_losses[i] /= client_num
            client_loss_dict[client_num] = average_n_elements(avg_losses, n=max(1, math.ceil(len(avg_losses) / 30)))
        else:
            client_loss_dict[client_num] = None
    return client_loss_dict


def get_aggregated_server_loss_convergence(model, version):
    files = get_all_log_files(log_dir)
    client_loss_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        target_files = list(
            filter(lambda x: model in x and version in x and 'training_metrics' in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        # for f in target_files:
        #     print(f)
        for file in target_files:
            losses = _get_server_loss_convergence(file)
            if losses != []:
                client_loss_dict[client_num] = losses
            else:
                client_loss_dict[client_num] = None
    return client_loss_dict

def get_aggregated_server_time(model, version):
    files = get_all_log_files(log_dir)
    client_loss_dict = {}
    for client_num in [1, 2, 3,4, 8, 16,24, 32, 48,64]:
        target_files = list(
            filter(lambda x: model in x and version in x and 'training_metrics' in x and x.split('/')[5] == f'client_number_{client_num}', files)
        )
        for f in target_files:
            print(f)
        for file in target_files:
            times = _get_server_time(file)
            if times != []:
                client_loss_dict[client_num] = times
            else:
                client_loss_dict[client_num] = None
    return client_loss_dict

import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', '-M', type=str, default='qwen3-0.6b')
    argparser.add_argument('--version', '-V', type=str, default='v2')
    argparser.add_argument('--dataset', '-DS', type=str, default='gsm8k')
    argparser.add_argument('--epoch_time', '-T', action='store_true')
    argparser.add_argument('--max_mem', '-MM', action='store_true')
    argparser.add_argument('--loss', '-L', action='store_true')
    argparser.add_argument('--aggragate_time', '-AT', action='store_true')
    args = argparser.parse_args()
    # log_dir = './log/meta-llama' if 'llama' in args.model else './log/qwen'
    log_dir=f'./log/loss/gsm8k'
    if args.dataset == 'ablation':
        #do ablation experiment
        print(f'analysis for model: {args.model}, version: {args.version}, ablation experiment')
        if args.epoch_time:
            print('epoch time analysis:')
            epoch_time = get_avg_epoch_time_ablation(args.model, args.version)
            for k, v in epoch_time.items():
                print(f'client_num: {k}, avg_epoch_time: {v} s')
        if args.max_mem:
            print('max mem analysis:')
            max_mem = get_max_mem_alloc_ablation(args.model, args.version)
            for k, v in max_mem.items():
                print(f'client_num: {k}, max_mem: {v} GB')
        if args.loss:
            print('client loss analysis:')
            client_loss = get_aggregated_server_loss_convergence(args.model, args.version)
            for k, v in client_loss.items():
                print(f'client_num: {k}, avg_client_loss:')
                if v is not None:
                    for l in v:
                        print(f'{l:.4f}')
    else:
        print(f'analysis for model: {args.model}, version: {args.version}, dataset: {args.dataset}')
        if args.epoch_time:
            print('epoch time analysis:')
            epoch_time = get_avg_epoch_time(args.model, args.version)
            for k, v in epoch_time.items():
                print(f'client_num: {k}, avg_epoch_time: {v} s')
        if args.max_mem:
            print('max mem analysis:')
            max_mem = get_max_mem_alloc(args.model, args.version)
            for k, v in max_mem.items():
                print(f'client_num: {k}, max_mem: {v} GB')
        if args.loss:
            print('client loss analysis:')
            client_loss = get_aggregated_server_loss_convergence(args.model, args.version)
            for k, v in client_loss.items():
                print(f'client_num: {k}, avg_client_loss:')
                if v is not None:
                    for l in v:
                        print(f'{l:.4f}')
        if args.aggragate_time:
            print('server time analysis:')
            server_time = get_aggregated_server_time(args.model, args.version)
            for k, v in server_time.items():
                print(f'client_num: {k}, avg_server_time:')
                if v is not None:
                    for l in v:
                        print(f'{l}')
