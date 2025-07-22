import os
import math

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
    with open(fp, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        if 'Finished' in last_line:
            start = last_line.index('train epoch time: ')
            end = start + len('train epoch time: ') + 7
            return last_line[start + len("train epoch time: ") : end]
        return None


def _get_mem_alloc(fp: str):
    max_mem = 0
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split('|')
            # print(values[1].strip())
            try:
                max_mem = max(max_mem, float(values[1].strip()))
            except Exception as e:
                continue
    return max_mem


def _get_server_loss_convergence(fp: str):
    losses = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Aggregated client models finished' in line:
                start = line.index('Aggregated client models finished')
                end = start + len('Aggregated client models finished, avg loss: ') + 6
                l = line[start + len("Aggregated client models finished, avg loss: ") : end]
                losses.append(float(l))
    return losses


def _get_client_loss_convergence(fp: str):
    losses = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'train batch' in line:
                start = line.index('loss: ')
                end = start + len('loss: ') + 6
                l = line[start + len("loss: ") : end]
                losses.append(float(l))
    return losses


def get_avg_epoch_time(model, version):
    files = get_all_log_files(log_dir)
    epoch_time_dict = {}
    for client_num in [1, 2, 4, 6, 8, 16, 24]:
        avg_time = 0
        target_files = list(
            filter(lambda x: model in x and version in x and 'server' not in x and x.split('/')[4] == f'client_number_{client_num}', files)
        )
        # for f in target_files:
        #     print(f)
        for file in target_files:
            epoch_time = _get_epoch_time(file)
            if epoch_time is not None:
                avg_time += float(epoch_time)
        if len(target_files) > 0:
            avg_time /= client_num
            epoch_time_dict[client_num] = f'{avg_time:.2f}'
        else:
            epoch_time_dict[client_num] = None
    return epoch_time_dict


def get_max_mem_alloc(model, version):
    files = get_all_log_files(log_dir)
    max_mem_dict = {}
    for client_num in [1, 2, 4, 6, 8, 16, 24]:
        max_mem = 0
        target_file = list(
            filter(lambda x: model in x and version in x and 'training_metrics' in x and x.split('/')[4] == f'client_number_{client_num}', files)
        )
        if len(target_file) > 0:
            max_mem = _get_mem_alloc(target_file[0])
            max_mem_dict[client_num] = f'{max_mem:.2f}'
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
    for client_num in [1, 2, 4, 6, 8, 16, 24]:
        avg_losses = []
        target_files = list(
            filter(lambda x: model in x and version in x and 'server' not in x and x.split('/')[4] == f'client_number_{client_num}', files)
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
            client_loss_dict[client_num] = average_n_elements(avg_losses, n=max(1, math.ceil(len(avg_losses) / 15)))
        else:
            client_loss_dict[client_num] = None
    return client_loss_dict


import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', '-M', type=str, default='qwen3-0.6b')
    argparser.add_argument('--version', '-V', type=str, default='v2')
    argparser.add_argument('--epoch_time', '-T', action='store_true')
    argparser.add_argument('--max_mem', '-MM', action='store_true')
    argparser.add_argument('--loss', '-L', action='store_true')
    args = argparser.parse_args()
    log_dir = './log'
    print(f'analysis for model: {args.model}, version: {args.version}')
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
        client_loss = get_client_loss_convergence(args.model, args.version)
        for k, v in client_loss.items():
            print(f'client_num: {k}, avg_client_loss:')
            if k is not None:
                for l in v:
                    print(f'{l:.4f}')
