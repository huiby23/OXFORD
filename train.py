import os
import numpy as np
import argparse
import torch


from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from models.UNet import Dual_UNet
from utils.loss import Point_Matching_Loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
random.seed(1)
from eval import Drift_rate_eval
# torch.cuda.set_per_process_memory_fraction(0.5)
def Args():
    parser = argparse.ArgumentParser(description="settings")
    # dataset
    parser.add_argument("--data_path", default='data/oxford_radar')
    parser.add_argument("--model_save_path", default='check_points')
    parser.add_argument("--logs_path", default='logs')
    parser.add_argument("--batch_size", default=6, type=int)
    # optimizer, default Adam
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--total_epoch", default=400, type=int)
    parser.add_argument("--img_sz", default=448, type=int)
    
    args = parser.parse_args()
    return args

def main():
    
    #init
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)

    #loss
    pm_loss = Point_Matching_Loss()
    best_val_loss = float("inf")
    
    # dataloader
    train_set = CustomDataset(args.data_path,args.img_sz,mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_set = CustomDataset(args.data_path,args.img_sz,mode="test")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.logs_path, 'logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    # 预测出的相对位姿变换的list，长度为epoch
    est_pose_tran_list = []  

    #train
    for epoch in range(args.total_epoch):

        print(f"EPOCH: {epoch}/{args.total_epoch}")
        model.train()
        total_loss = 0

        # 根据学习的关键点，预测出的相对位姿变换
        est_pose_tran = []  
        
        for batch_idx, (im1, im2, pos_trans) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", ncols=100)):
            im1, im2,pos_trans = im1.to(device), im2.to(device), pos_trans.to(device)
            im = torch.cat([im1,im2],dim=0)
            optimizer.zero_grad()
            locations_map, scores_map, descriptors_map = model(im)
            B=int(locations_map.shape[0]/2)
            locations_map1, scores_map1, descriptors_map1 = locations_map[0:B,:,:,:], scores_map[0:B,:,:,:], descriptors_map[0:B,:,:,:]
            _, scores_map2, descriptors_map2 = locations_map[B:,:,:,:], scores_map[B:,:,:,:], descriptors_map[B:,:,:,:]
            loss = pm_loss(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,pos_trans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  
            optimizer.step()

            total_loss += loss.item()

            # odometry estimation
            _, _, est_translation, est_rotation = pm_loss.match(locations_map1, scores_map1, descriptors_map1, 
                                                                          scores_map2, descriptors_map2,threshold=0.01)
            
            # 保存根据学习的关键点预测出的相对位姿变换
            est_translation = est_translation.cpu().numpy()   # (B, 2, 2)
            est_rotation = est_rotation.cpu().numpy()         # (B, 2)

            est_rotation = est_rotation.reshape(-1, 2, 1)  # (B, 2, 1)

            # 拼接得到完整的SE2位姿变换矩阵 (B, 2, 3)
            for k in range(B):
                R = est_rotation[k]
                T = est_translation[k]
                est_pose = np.concatenate([R, T], axis=-1)  # (2, 3)

                est_pose_tran.append(est_pose)
        
        avg_train_loss = total_loss / len(train_loader)
        lr_scheduler.step(avg_train_loss)  
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, LR: {optimizer.param_groups[0]['lr']}")

        # Log to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (im1, im2, pos_trans,_) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch}", ncols=100)):
                im1, im2, pos_trans = im1.to(device), im2.to(device), pos_trans.to(device)
                im = torch.cat([im1,im2],dim=0)
                locations_map, scores_map, descriptors_map = model(im)
                B=int(locations_map.shape[0]/2)
                locations_map1, scores_map1, descriptors_map1 = locations_map[0:B,:,:,:], scores_map[0:B,:,:,:], descriptors_map[0:B,:,:,:]
                _, scores_map2, descriptors_map2 = locations_map[B:,:,:,:], scores_map[B:,:,:,:], descriptors_map[B:,:,:,:]
                val_loss = pm_loss(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2, pos_trans)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # Log validation loss to TensorBoard
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

        torch.save(model.state_dict(), "{}/model_last.pth".format(args.model_save_path))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "{}/model_best.pth".format(args.model_save_path))
        
        
        # Save the estimated pose transformation for drift rate evaluation
        est_pose_tran_list.append(est_pose_tran)
    
    # ---------- drift rate evaluation ----------
    val_result_dir = os.path.join(args.model_save_path, 'train_drift_rate')
    if not os.path.exists(val_result_dir):
        os.makedirs(val_result_dir)
    
    drift_rate_val = Drift_rate_eval()
    drift_rate_val(args.data_path, est_pose_tran_list[epoch-1], val_result_dir, epoch)


if __name__ == '__main__':
    main()