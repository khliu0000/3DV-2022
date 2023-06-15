import torch

# define losses
def voxel_loss(voxel_src, voxel_tgt):
    pred = voxel_src.view(-1, 1).float().clamp(1e-7, 1-1e-7)
    gt = voxel_tgt.view(-1, 1).float()
    prob_loss = -1 * ((gt*(pred+1e-7).log())+((1-gt)*(1-pred+1e-7).log())).mean()
    return prob_loss

def chamfer_loss(point_cloud_src, point_cloud_tgt, each_batch=False):
    loss1 = pts_dist(point_cloud_src, point_cloud_tgt)
    loss2 = pts_dist(point_cloud_tgt, point_cloud_src)
    loss_chamfer = loss1 + loss2
    if each_batch:
        return loss_chamfer
    else:
        return loss_chamfer.mean()

def pts_dist(pts1, pts2):
    pts1 = pts1[:, :, None, :]
    pts2 = pts2[:, None, :, :]
    min_dist = pts1-pts2
    min_dist = min_dist * min_dist
    min_dist = min_dist.sum(dim=3).sqrt()
    min_dist = min_dist * min_dist
    min_dist = min_dist.min(dim=2).values.mean(dim=1)
    return min_dist

class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, w1=1.0, w2=1.0, each_batch=False):
        self.check_parameters(points1)
        self.check_parameters(points2)

        diff = points1[:, :, None, :] - points2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist1 = dist
        dist2 = torch.transpose(dist, 1, 2)

        dist1 = torch.sqrt(dist1)**2
        dist2 = torch.sqrt(dist2)**2

        dist_min1, indices1 = torch.min(dist1, dim=2)
        dist_min2, indices2 = torch.min(dist2, dim=2)

        loss1 = dist_min1.mean(1)
        loss2 = dist_min2.mean(1)
        
        loss = w1 * loss1 + w2 * loss2

        if not each_batch:
            loss = loss.mean()

        return loss

    @staticmethod
    def check_parameters(points: torch.Tensor):
        assert points.ndimension() == 3  # (B, N, 3)
        assert points.size(-1) == 3