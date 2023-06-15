from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
# from pytorch3d.utils import ico_sphere
# import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, cfg):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[cfg.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if cfg.dtype == "voxel":
            self.decoder = VoxelDecoder(cfg.latent_size)
        elif cfg.dtype == "point":
            self.n_point = cfg.n_points
            self.decoder = PointDecoder(cfg.n_points, cfg.latent_size)     

    def forward(self, images, cfg):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize)
        _, C, H, W = encoded_feat.shape  # (10, 512, 1, 1)
        encoded_feat = encoded_feat.squeeze(-1).squeeze(-1)

        # call decoder
        if cfg.dtype == "voxel":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred
        elif cfg.dtype == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred       


class PointDecoder(nn.Module):
    def __init__(self, num_points, latent_size):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc0 = nn.Linear(latent_size, 100)
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()    
        
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, self.num_points, 3)
        return x


class VoxelDecoder(nn.Module):
    def __init__(self, latent_size):
        self.latent_size = latent_size
        super(VoxelDecoder, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU()
        )
        # (10, 4096) reshape to (10, 512, 2, 2, 2)  (size-1)stride-2padding+1+(k-1)+op
        self.layer3 = torch.nn.Sequential(
            nn.ConvTranspose3d(4096//8, 256, kernel_size=4, stride=2, padding=1),  # (10, 256, 4, 4, 4)
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # (10, 128, 8, 8, 8)
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # (10, 64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            nn.ConvTranspose3d(64, 16, kernel_size=4, stride=2, output_padding=1, padding=1),  # (10, 16, 33, 33, 33)
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1),  # (10, 1, 33, 33, 33)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(batch_size, -1, 2, 2, 2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(batch_size, 33, 33, 33)
        return x