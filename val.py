import os
import cv2
import json
import torch
import pickle
import argparse
import numpy as np


from networks.Oxford_Radar import Oxford_Radar
from utils.dataloader import get_dataloaders
from utils.utils import computeMedianError, computeKittiMetrics, load_icra21_results, save_in_yeti_format_new, get_transform2
from utils.utils import plot_sequences
from time import time

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

# 根据score筛选关键点并进行绘制
score_threshold = 0.1

def get_folder_from_file_path(path):
    elems = path.split('/')
    newpath = ""
    for j in range(0, len(elems) - 1):
        newpath += elems[j] + "/"
    return newpath


def add_text_to_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # 白色文字
    thickness = 2
    position = (10, 30)  # 左上角位置
    image_with_text = cv2.putText(image.copy(), text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return image_with_text


def filter_keypoints_by_weight(src_coords, tgt_coords, weight_scores, threshold = 0.5):
    """
    根据权重筛选关键点
    参数:
        src_coords: (B, N, 2) 源关键点坐标
        tgt_coords: (B, N, 2) 目标关键点坐标
        weight_scores: (B, 1, H, W) 权重分数图
        threshold: 权重阈值
    返回:
        valid_src: (B, M, 2) 筛选后的源关键点
        valid_tgt: (B, M, 2) 筛选后的目标关键点
    """
    valid_src = []
    valid_tgt = []
    
    batch_size = src_coords.shape[0]
    for b in range(batch_size):
        batch_src = []
        batch_tgt = []
        weights = weight_scores[b, 0]  # (H, W)
        
        for (x1, y1), (x2, y2) in zip(src_coords[b], tgt_coords[b]):
            x1_int = int(round(x1))
            y1_int = int(round(y1))
            
            # 检查坐标是否在有效范围内
            if (0 <= y1_int < weights.shape[0]) and (0 <= x1_int < weights.shape[1]):
                if weights[y1_int, x1_int] > threshold:
                    batch_src.append([x1, y1])
                    batch_tgt.append([x2, y2])
        
        valid_src.append(np.array(batch_src))
        valid_tgt.append(np.array(batch_tgt))
    
    return np.array(valid_src), np.array(valid_tgt)


def filter_keypoints(src_coords, tgt_coords, weight_scores, num_points = 30):
    """
    筛选出分数前30的关键点
    参数:
        src_coords: (B, N, 2) 源关键点坐标
        tgt_coords: (B, N, 2) 目标关键点坐标
        weight_scores: (B, 1, H, W) 权重分数图
    返回:
        valid_src: (B, M, 2) 筛选后的源关键点
        valid_tgt: (B, M, 2) 筛选后的目标关键点
    """
    valid_src = []
    valid_tgt = []
    
    batch_size = src_coords.shape[0]
    for b in range(batch_size):
        batch_src = []
        batch_tgt = []
        weights = weight_scores[b, 0]  # (H, W)
        src_x = src_coords[b, :, 0]  # (N,)
        src_y = src_coords[b, :, 1]  # (N,)
        src_weight = weights[src_y.astype(int), src_x.astype(int)]  # (N,)

        tgt_x = tgt_coords[b, :, 0]  # (N,)
        tgt_y = tgt_coords[b, :, 1]  # (N,)
        tgt_weight = weights[tgt_y.astype(int), tgt_x.astype(int)]  # (N,)

        sum_weight = src_weight + tgt_weight    # (N,)
        index = sorted(range(len(sum_weight)), key=lambda i: sum_weight[i], reverse=True)[:num_points]  # 获取前30个索引
        
        for k in range(num_points):
            x1 = int(round(src_x[index[k]]))
            y1 = int(round(src_y[index[k]]))
            x2 = int(round(tgt_x[index[k]]))
            y2 = int(round(tgt_y[index[k]]))

            batch_src.append([x1, y1])
            batch_tgt.append([x2, y2])
        
        valid_src.append(np.array(batch_src))
        valid_tgt.append(np.array(batch_tgt))
    
    return np.array(valid_src), np.array(valid_tgt)


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/steam.json', type=str, help='config file path')
    parser.add_argument('--pretrain', default=None, type=str, help='pretrain checkpoint path')
    parser.add_argument('--val_save_path', default='results', type=str, help='validation save directory')
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

    if config['model'] == 'Oxford_Radar':
        model = Oxford_Radar(config).to(config['gpuid'])
    assert(args.pretrain is not None)


    # load model
    checkpoint = torch.load(args.pretrain, map_location=torch.device(config['gpuid']))
    failed = False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        print(e)
        failed = True
    if failed:
        model.load_state_dict(checkpoint, strict=False)
    

    # img save path
    if not os.path.exists(args.val_save_path):
        os.makedirs(args.val_save_path)

    if not os.path.exists(os.path.join(args.val_save_path, 'combine')):
        os.makedirs(os.path.join(args.val_save_path, 'combine'), exist_ok=True)
    
    if not os.path.exists(os.path.join(args.val_save_path, 'val')):
        os.makedirs(os.path.join(args.val_save_path, 'val'), exist_ok=True)


    # validation
    model.eval()

    T_gt_ = []
    T_pred_ = []
    t_errs = []
    r_errs = []
    time_used_ = []
    img_width = config['cart_pixel_width']

    for seq_num in seq_nums:
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
        for batchi, batch in enumerate(test_loader):
            ts = time()
            if (batchi + 1) % config['print_rate'] == 0:
                print('Eval Batch {} / {}: {:.2}s'.format(batchi, len(test_loader), np.mean(time_used[-config['print_rate']:])))
            with torch.no_grad():
                try:
                    out = model(batch)
                except RuntimeError as e:
                    print(e)
                    continue
            
            # keypoint information save
            input_data = batch['data'].cpu().numpy()                # (B, 1, H, W)
            detector_scores = out['detector_scores'].cpu().numpy()  # (B, 1, H, W)
            weight_scores = out['scores'].cpu().numpy()             # (B, 1, H, W)
            src_coords = out['src'].cpu().numpy()                   # (B, num_points, 2)
            tgt_coords = out['tgt'].cpu().numpy()                   # (B, num_points, 2)
            sfm_val = out['soft_match_vals'].cpu().numpy()          # (B, num_points, HW)
            
            # 根据score筛选关键点
            valid_src_coords, valid_tgt_coords = filter_keypoints_by_weight(src_coords, tgt_coords, weight_scores, score_threshold)
            # valid_src_coords, valid_tgt_coords = filter_keypoints(src_coords, tgt_coords, weight_scores, 30)

            # config['model'] == 'Oxford_Radar':
            T_gt.append(batch['T_21'][0].numpy().squeeze())
            R_pred_ = out['R'][0].detach().cpu().numpy().squeeze()
            t_pred_ = out['t'][0].detach().cpu().numpy().squeeze()
            T_pred.append(get_transform2(R_pred_, t_pred_))
            
            # print('T_gt:\n{}'.format(T_gt[-1]))
            # print('T_pred:\n{}'.format(T_pred[-1]))
            time_used.append(time() - ts)
            if 'timestamps' in batch:
                timestamps.append(batch['timestamps'][0].numpy())

            # 遍历batch中的每个样本
            batch_size = src_coords.shape[0]
            for i in range(batch_size):
                # 处理输入图像
                img = input_data[i, 0]
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                valid_img = img.copy()
                img = add_text_to_image(img, 'Input')
                
                # 绘制匹配点
                src_points = src_coords[i]
                tgt_points = tgt_coords[i]
                for (x1, y1), (x2, y2) in zip(src_points, tgt_points):
                    x1, y1 = int(round(x1)), int(round(y1))
                    x2, y2 = int(round(x2)), int(round(y2))
                    cv2.circle(img, (x1, y1), 3, (0,255,0), -1)
                    cv2.circle(img, (x2, y2), 3, (0,0,255), -1)
                    cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 1)
                
                valid_img = add_text_to_image(valid_img, 'Valid Points')
        
                # 绘制筛选后的关键点
                valid_src_points = valid_src_coords[i]
                valid_tgt_points = valid_tgt_coords[i]
                for (x1, y1), (x2, y2) in zip(valid_src_points, valid_tgt_points):
                    x1, y1 = int(round(x1)), int(round(y1))
                    x2, y2 = int(round(x2)), int(round(y2))
                    cv2.circle(valid_img, (x1, y1), 3, (0,255,0), -1)
                    cv2.circle(valid_img, (x2, y2), 3, (0,0,255), -1)
                    cv2.line(valid_img, (x1, y1), (x2, y2), (255,0,0), 1)
                    
                # 处理detector_scores_1
                det_1 = detector_scores[i, 0]
                det_1 = cv2.normalize(det_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                det_img_1 = cv2.cvtColor(det_1, cv2.COLOR_GRAY2BGR)
                det_img_1 = add_text_to_image(det_img_1, 'Detector 1')
                
                # 处理detector_scores_2
                det_2 = detector_scores[i + 1, 0]
                det_2 = cv2.normalize(det_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                det_img_2 = cv2.cvtColor(det_2, cv2.COLOR_GRAY2BGR)
                det_img_2 = add_text_to_image(det_img_2, 'Detector 2')

                # 处理weight_scores_1
                weight_1 = weight_scores[i, 0]
                weight_1 = cv2.normalize(weight_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                weight_img_1 = cv2.cvtColor(weight_1, cv2.COLOR_GRAY2BGR)
                weight_img_1 = add_text_to_image(weight_img_1, 'Weight 1')

                # 处理weight_scores_2
                weight_2 = weight_scores[i + 1, 0]
                weight_2 = cv2.normalize(weight_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                weight_img_2 = cv2.cvtColor(weight_2, cv2.COLOR_GRAY2BGR)
                weight_img_2 = add_text_to_image(weight_img_2, 'Weight 2')

                # source帧关键点坐标
                ps_img = np.zeros((img_width, img_width), dtype=np.uint8)
                x, y = src_coords[i,:,0], src_coords[i,:,1]
                ps_img[y.astype(int), x.astype(int)] = 255
                ps_img = (ps_img).astype(np.uint8)
                ps_img = cv2.cvtColor(ps_img, cv2.COLOR_GRAY2BGR)
                ps_img = add_text_to_image(ps_img, 'Ps location')

                # destination帧关键点坐标
                pd_img = np.zeros((img_width, img_width), dtype=np.uint8)
                x, y = tgt_coords[i,:,0], tgt_coords[i,:,1]
                pd_img[y.astype(int), x.astype(int)] = 255
                pd_img = (pd_img).astype(np.uint8)
                pd_img = cv2.cvtColor(pd_img, cv2.COLOR_GRAY2BGR)
                pd_img = add_text_to_image(pd_img, 'Pd location')

                # 统一尺寸
                det_img_1 = cv2.resize(det_img_1, (img_width, img_width))
                det_img_2 = cv2.resize(det_img_2, (img_width, img_width))
                weight_img_1 = cv2.resize(weight_img_1, (img_width, img_width))
                weight_img_2 = cv2.resize(weight_img_2, ((img_width, img_width)))

                # 拼接并保存
                row_1 = cv2.hconcat([det_img_1, det_img_2, weight_img_1, weight_img_2])
                row_2 = cv2.hconcat([ps_img, pd_img, img, valid_img])
                combined = cv2.vconcat([row_1, row_2])
                seq_name = seq_names[0]  # 获取当前序列名
                save_name = f"{seq_name}_batch{batchi}_sample{i}.png"
                cv2.imwrite(os.path.join(args.val_save_path, 'combine', save_name), combined)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"input_{save_name}"), img)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"det_1_{save_name}"), det_img_1)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"det_2_{save_name}"), det_img_2)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"weight_1_{save_name}"), weight_img_1)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"weight_2_{save_name}"), weight_img_2)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"ps_{save_name}"), ps_img)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"pd_{save_name}"), pd_img)
                cv2.imwrite(os.path.join(args.val_save_path, 'val', f"valid_{save_name}"), valid_img)
        
        T_gt_.extend(T_gt)
        T_pred_.extend(T_pred)
        time_used_.extend(time_used)

        # print('T_gt:\n', T_gt)
        # print('T_pred:\n', T_pred)
        t_err, r_err = computeKittiMetrics(T_gt, T_pred, [len(T_gt)])
        
        print('SEQ: {} : {}'.format(seq_num, seq_names[0]))
        print('KITTI t_err: {} %'.format(t_err))
        print('KITTI r_err: {} deg/m'.format(r_err))
        
        t_errs.append(t_err)
        r_errs.append(r_err)

        save_in_yeti_format_new(T_gt, T_pred, [len(T_gt)], seq_names, root)
        pickle.dump([T_gt, T_pred, timestamps], open(root + 'odom' + seq_names[0] + '.obj', 'wb'))
        T_icra = None
        if config['dataset'] == 'oxford':
            if config['compare_yeti']:
                T_icra = load_icra21_results('./results/icra21/', seq_names, seq_lens)
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
