import numpy as np
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from models.UNet import Dual_UNet
from utils.loss import Point_Matching_Loss
from tqdm import tqdm
# torch.cuda.set_per_process_memory_fraction(0.5)
def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model_path", default="check_points/model_best.pth")
    # dataset
    parser.add_argument("--data_path", default='data/oxford_radar')
    parser.add_argument("--num_keypoints", default=400, type=int)
    parser.add_argument("--img_sz", default=448, type=int)
    parser.add_argument("--val_save_path", default='results/val')
    parser.add_argument("--batch_size", default=8)
    args = parser.parse_args()
    return args

def main():
    
    #init
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_UNet().to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    #loss
    pm_loss = Point_Matching_Loss()
    
    # dataloader
    val_set = CustomDataset(args.data_path,args.img_sz,mode="test")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)


    if not os.path.exists(args.val_save_path):
        os.makedirs(args.val_save_path)
    #train
    
    model.eval()

    with torch.no_grad():
        for batch_idx, (im1, im2, _,imgs_name) in enumerate(tqdm(val_loader, desc=f"inferencing", ncols=100)):
            im1, im2= im1.to(device), im2.to(device)
            im = torch.cat([im1,im2],dim=0)
            locations_map, scores_map, descriptors_map = model(im)
            # locations_map1, scores_map1, descriptors_map1 = locations_map[0:args.batch_size,:,:,:], scores_map[0:args.batch_size,:,:,:], descriptors_map[0:args.batch_size,:,:,:]
            # locations_map2, scores_map2, descriptors_map2 = locations_map[args.batch_size:,:,:,:], scores_map[args.batch_size:,:,:,:], descriptors_map[args.batch_size:,:,:,:]
            B=int(im2.shape[0])
            locations_map1, scores_map1, descriptors_map1 = locations_map[0:B,:,:,:], scores_map[0:B,:,:,:], descriptors_map[0:B,:,:,:]
            locations_map2, scores_map2, descriptors_map2 = locations_map[B:,:,:,:], scores_map[B:,:,:,:], descriptors_map[B:,:,:,:]



            matched_points = pm_loss.match(locations_map1, scores_map1, descriptors_map1, scores_map2, descriptors_map2,threshold=0.3)
            
            for i in range(B):

                loc=locations_map1[i, 0].cpu().numpy()
                loc = cv2.normalize(loc, None, 0, 255, cv2.NORM_MINMAX)
                loc = (loc).astype(np.uint8)# 归一化到 0-255 范围
                loc = cv2.cvtColor(loc, cv2.COLOR_GRAY2BGR)
                # cv2.imwrite('loc.png', loc)

                scores=scores_map1[i, 0].cpu().numpy()
                # scores = (scores * 255).astype(np.uint8)  # 归一化到 0-255 范围
                scores = cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)
                scores = scores.astype(np.uint8)

                scores = cv2.cvtColor(scores, cv2.COLOR_GRAY2BGR)
                # cv2.imwrite('scores.png', scores)
                
                img_name = imgs_name[i]+'.png'
                img = im2[i, 0].cpu().numpy()  # 提取单通道图像，并转换为 NumPy
                img = (img * 255).astype(np.uint8)  # 归一化到 0-255 范围
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为三通道 BGR 图像

                # cv2.imwrite('img.png', img)
                
                ps_i = np.array(matched_points[i][0].cpu())  # 提取 matched_ps
                pd_i = np.array(matched_points[i][1].cpu())  # 提取 matched_pd

                # 画匹配点
                for (x1, y1), (x2, y2) in zip(ps_i, pd_i):
                    cv2.circle(img, (int(x1), int(y1)), 3, (0, 255, 0), -1)  # 绿色点 (ps)
                    cv2.circle(img, (int(x2), int(y2)), 3, (0, 0, 255), -1)  # 红色点 (pd)
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)  # 连接线 (蓝色)

                save_path = os.path.join(args.val_save_path,img_name)

                combined_image = cv2.hconcat([loc, scores, img])
                
                cv2.imwrite('results/combine/{}'.format(img_name), combined_image)
                cv2.imwrite(save_path, img)

            # print(f"val_f1_score: {total_f1_score / total_samples}")


if __name__ == '__main__':
    main()