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
    parser.add_argument("--model_path", default="resnet18")
    # dataset
    parser.add_argument("--data_path", default='data/oxford_radar')
    # optimizer, default Adam
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
    val_set = CustomDataset(args.data_path,args.img_sz,mode="test")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    #train
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (im1, im2, pos_trans) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch}", ncols=100)):
            im1, im2,pos_trans = im1.to(device), im2.to(device), pos_trans.to(device)
            im = torch.cat([im1,im2],dim=0)
            locations_map, scores_map, descriptors_map = model(im)
            locations_map1, scores_map1, descriptors_map1 = locations_map[0:args.batch_size,:,:,:], scores_map[0:args.batch_size,:,:,:], descriptors_map[0:args.batch_size,:,:,:]
            locations_map2, scores_map2, descriptors_map2 = locations_map[args.batch_size:,:,:,:], scores_map[args.batch_size:,:,:,:], descriptors_map[args.batch_size:,:,:,:]
            matched_points = pm_loss.eval(locations_map1, scores_map1, descriptors_map1,locations_map2, scores_map2, descriptors_map2,threshold=0.5)



            # print(f"val_f1_score: {total_f1_score / total_samples}")


if __name__ == '__main__':
    main()