import os
import cv2
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm


from networks.Oxford_Radar import Oxford_Radar
from utils.dataloader import get_dataloaders
from utils.utils import computeMedianError, computeKittiMetrics, load_icra21_results, save_in_yeti_format_new, get_transform2
from utils.utils import plot_sequences
from time import time

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    args = parser.parse_args()
    return args


def main():
    # initialize
    torch.set_num_threads(8)
    args = Args()

    with open(args.config) as f:
        config = json.load(f)
    root = get_folder_from_file_path(args.pretrain)
    seq_nums = config['test_split']

    model = Oxford_Radar(config).to(config['gpuid'])
    assert(args.pretrain is not None)

    # load model
    checkpoint = torch.load(args.pretrain, map_location=torch.device(config['gpuid']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    
    # validation
    model.eval()

    T_gt_ = []
    T_pred_ = []
    t_errs = []
    r_errs = []
    time_used_ = []

    for seq_num in seq_nums:
        torch.cuda.empty_cache()    # 清理GPU缓存
        time_used = []
        T_gt = []
        T_pred = []
        timestamps = []
        config['test_split'] = [seq_num]
        
        # config['dataset'] == 'oxford'
        _, _, test_loader = get_dataloaders(config)

        seq_lens = test_loader.dataset.seq_lens
        print(seq_lens)
        seq_names = test_loader.dataset.sequences
        print('Evaluating sequence: {} : {}'.format(seq_num, seq_names[0]))
        
        with torch.no_grad():
            for batchi, batch in enumerate(tqdm(test_loader, desc=f"Value {seq_names[0]}", unit="batch")):
                ts = time()
                with torch.no_grad():
                    try:
                        out = model(batch)
                    except RuntimeError as e:
                        print(e)
                        continue
                
                T_gt.append(batch['T_21'][0].numpy().squeeze())
                R_pred_ = out['R'][0].detach().cpu().numpy().squeeze()
                t_pred_ = out['t'][0].detach().cpu().numpy().squeeze()
                T_pred.append(get_transform2(R_pred_, t_pred_))
                
                time_used.append(time() - ts)
                if 'timestamps' in batch:
                    timestamps.append(batch['timestamps'][0].numpy())


        T_gt_.extend(T_gt)
        T_pred_.extend(T_pred)
        time_used_.extend(time_used)

        t_err, r_err = computeKittiMetrics(T_gt, T_pred, [len(T_gt)])
        
        print('SEQ: {} : {}'.format(seq_num, seq_names[0]))
        print('KITTI t_err: {} %'.format(t_err))
        print('KITTI r_err: {} deg/m'.format(r_err))
        
        t_errs.append(t_err)
        r_errs.append(r_err)

        T_icra = None
        # save_in_yeti_format_new(T_gt, T_pred, [len(T_gt)], seq_names, root)
        # pickle.dump([T_gt, T_pred, timestamps], open(root + 'odom' + seq_names[0] + '.obj', 'wb'))
        # if config['dataset'] == 'oxford':
        #     if config['compare_yeti']:
        #         T_icra = load_icra21_results('./results/icra21/', seq_names, seq_lens)
        fname = root + seq_names[0] + '.pdf'
        plot_sequences(T_gt, T_pred, [len(T_gt)], returnTensor=False, T_icra=T_icra, savePDF=True, fnames=[fname])

    print('time_used: {}'.format(sum(time_used_) / len(time_used_)))
    results = computeMedianError(T_gt_, T_pred_)
    with open('errs.obj', 'wb') as f:
        pickle.dump([results[-2], results[-1]], f)
    print('dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(results[0], results[1], results[2], results[3]))

    t_err = np.mean(t_errs)
    r_err = np.mean(r_errs)
    print('Average KITTI metrics over all test sequences:')
    print('KITTI t_err: {} %'.format(t_err))
    print('KITTI r_err: {} deg/m'.format(r_err))



if __name__ == '__main__':
    main()
