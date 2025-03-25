import numpy as np
import argparse
import torch


from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from models.UNet import Dual_UNet
from utils.loss import Point_Matching_Loss
from tqdm import tqdm
# torch.cuda.set_per_process_memory_fraction(0.5)
def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet18")
    # dataset
    parser.add_argument("--data_path", default='data/oxford_radar')
    parser.add_argument("--batch_size", default=8, type=int)
    # optimizer, default Adam
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=50, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--num_keypoints", default=400, type=int)
    parser.add_argument("--img_sz", default=448, type=int)
    args = parser.parse_args()
    return args

def main():
    
    #init
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_UNet().to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    #loss
    pm_loss = Point_Matching_Loss()
    
    # dataloader
    train_set = CustomDataset(args.data_path,args.img_sz,mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_set = CustomDataset(args.data_path,args.img_sz,mode="test")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    #train
    for epoch in range(args.total_epoch):

        print(f"EPOCH: {epoch}/{args.total_epoch}")
        model.train()
        total_loss = 0
        for batch_idx, (im1, im2, pos_trans) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", ncols=100)):
            im1, im2,pos_trans = im1.to(device), im2.to(device), pos_trans.to(device)
            im = torch.cat([im1,im2],dim=0)
            optimizer.zero_grad()
            locations_map, scores_map, descriptors_map = model(im)
            locations_map1, scores_map1, descriptors_map1 = locations_map[0:args.batch_size,:,:,:], scores_map[0:args.batch_size,:,:,:], descriptors_map[0:args.batch_size,:,:,:]
            _, scores_map2, descriptors_map2 = locations_map[args.batch_size:,:,:,:], scores_map[args.batch_size:,:,:,:], descriptors_map[args.batch_size:,:,:,:]
            loss = pm_loss(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,pos_trans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 防止梯度爆炸
            optimizer.step()

            total_loss += loss.item()
            # if batch_idx % 10 == 0:
            #     print(f"batch_idx: {batch_idx}, train_loss: {loss.item()}")
        
        # lr_scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        lr_scheduler.step(avg_train_loss)  # 依据 loss 调整学习率
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, LR: {optimizer.param_groups[0]['lr']}")

        # model.eval()
        # with torch.no_grad():
        #     total_samples, total_f1_score = 0, 0
        #     for batch_idx, (im1, im2, pos_trans) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch}", ncols=100)):
        #         im1, im2,pos_trans = im1.to(device), im2.to(device), pos_trans.to(device)
        #         im = torch.cat([im1,im2],dim=0)
        #         locations_map, scores_map, descriptors_map = model(im)
        #         locations_map1, scores_map1, descriptors_map1 = locations_map[0:args.batch_size,:,:,:], scores_map[0:args.batch_size,:,:,:], descriptors_map[0:args.batch_size,:,:,:]
        #         locations_map2, scores_map2, descriptors_map2 = locations_map[args.batch_size:,:,:,:], scores_map[args.batch_size:,:,:,:], descriptors_map[args.batch_size:,:,:,:]
                
        #     print(f"val_f1_score: {total_f1_score / total_samples}")


if __name__ == '__main__':
    main()