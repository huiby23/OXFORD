import os

from data_loader import Data_Preprocess_merged
from odometry_val import Evaluator


class Drift_rate_eval:
    def __init__(self):
        pass

    def __call__(self, data_path, est_pose_tran):
        self.run(data_path, est_pose_tran)

    def run(self, data_path, est_pose_tran): 
        # ---------- drift rate evaluation ---------- 

        dataset_path = data_path
        if not os.path.exists(dataset_path):
            assert IOError(f"Dataset path {dataset_path} does not exist.")
        print(f"Dataset path: {dataset_path}")

        data_preprocessor = Data_Preprocess_merged(dataset_path)

        # load gt_pose_tran data
        gt_pose_tran, _ = data_preprocessor.road_odometry_loader()

        # save gt_pose_tran data
        val_result_dir = '.\\val'
        if not os.path.exists(val_result_dir):
            os.makedirs(val_result_dir)

        gt_pose_tran_dir = os.path.join(val_result_dir, 'gt_pose_tran')
        if not os.path.exists(gt_pose_tran_dir):
            os.makedirs(gt_pose_tran_dir)

        # 行优先顺序写入
        with open(os.path.join(gt_pose_tran_dir, 'gt.txt'), 'w') as f:
            for matrix in gt_pose_tran:
                flattened = matrix.reshape(-1)
                line = ' '.join(map(str, flattened))
                f.write(line + '\n')

        # save estimated_pose_tran data
        est_pose_tran_dir = os.path.join(val_result_dir, 'est_pose_tran')
        if not os.path.exists(est_pose_tran_dir):
            os.makedirs(est_pose_tran_dir)

        # 行优先顺序写入
        with open(os.path.join(est_pose_tran_dir, 'result.txt'), 'w') as f:
            for matrix in est_pose_tran:
                flattened = matrix.reshape(-1)
                line = ' '.join(map(str, flattened))
                f.write(line + '\n')

        # compute translational and rotational error
        vo_eval = Evaluator()
        vo_eval.eval(gt_pose_tran_dir, est_pose_tran_dir)


if __name__ == '__main__':
    val = Drift_rate_eval()
    data_path = './data'
    val(data_path, est_pose_tran=None)