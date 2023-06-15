import time
import torch
from src.dataset import ShapeNetDB
from src.model import SingleViewto3D
import src.losses as losses
import numpy as np

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt


cd_loss = losses.chamfer_loss

def calculate_loss(predictions, ground_truth, cfg):
    if cfg.dtype == 'voxel':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif cfg.dtype == 'point':
        loss = cd_loss(predictions, ground_truth)
           
    return loss

@hydra.main(config_path="configs/", config_name="config.yml")
def evaluate_model(cfg: DictConfig):
    shapenetdb = ShapeNetDB(cfg.data_dir, cfg.dtype)

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(cfg)
    model.cuda()
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_loss = []

    if cfg.load_eval_checkpoint:
        checkpoint = torch.load(f'{cfg.base_dir}/checkpoint_{cfg.dtype}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    with torch.no_grad():
        max_iter = len(eval_loader)
        for step in range(start_iter, max_iter):
            iter_start_time = time.time()

            read_start_time = time.time()

            images_gt, ground_truth_3d, _ = next(eval_loader)
            images_gt, ground_truth_3d = images_gt.cuda(), ground_truth_3d.cuda()

            read_time = time.time() - read_start_time

            prediction_3d = model(images_gt, cfg)
            torch.save(prediction_3d.detach().cpu(), f'{cfg.base_dir}/pre_point_cloud.pt')

            loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()

            if (step % cfg.vis_freq) == 0:
                batch_size = images_gt.shape[0]
                fig = plt.figure(figsize=(8, 25))
                for i in range(batch_size):
                    print(f"plotting iter {i}")
                    a = fig.add_subplot(1*batch_size, 3, 1+i*3)
                    imgplot = plt.imshow(images_gt[i].cpu())
                    a.axis("off")
                    a.set_title("rgb input")
                    a = fig.add_subplot(1*batch_size, 3, 2+i*3, projection='3d')
                    gt = ground_truth_3d[i].cpu()
                    if cfg.dtype == 'point':
                        a.scatter(gt[:,0], gt[:,1], gt[:,2])
                        a.set_title("pointcloud gt")
                    elif cfg.dtype == 'voxel':
                        gt = gt.float()
                        print("processing gt")
                        a.voxels(gt)
                        a.set_title("voxel gt")
                    a.axis("off")
                    a = fig.add_subplot(1*batch_size, 3, 3+i*3, projection='3d')
                    pred = prediction_3d[i].cpu()
                    if cfg.dtype == 'point':
                        a.scatter(pred[:,0], pred[:,1], pred[:,2])
                        a.set_title("pointcloud predicted")
                    elif cfg.dtype == 'voxel':
                        pred = pred.ge(0.5).int()
                        print("processing prediction")
                        a.voxels(pred)
                        a.set_title("voxel predicted")
                    a.axis("off")
                print("##### saving_visualization #####")
                plt.savefig(f'{cfg.base_dir}vis/{step}_{cfg.dtype}.png', bbox_inches='tight')

            total_time = time.time() - start_time
            iter_time = time.time() - iter_start_time

            avg_loss.append(loss)

            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); eva_loss: %.6f" % (step, cfg.max_iter, total_time, read_time, iter_time, torch.tensor(avg_loss).mean()))

    print('Done!')

if __name__ == '__main__':
    evaluate_model()