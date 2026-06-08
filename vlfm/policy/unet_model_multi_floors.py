import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import datetime

class HistoryAwareUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base_channels=32): # in_channels=13
        """
        历史感知UNet架构
        :param in_channels: 总输入通道数
        :param out_channels: 输出通道数
        :param base_channels: 基础通道数
        """
        super(HistoryAwareUNet, self).__init__()
        
        # 输入预处理 - 分离不同类型输入
        self.input_preprocess = InputPreprocessor()
        
        # 编码器路径
        # =====================================
        # self.embed = nn.Conv2d(in_channels, base_channels*2, 1)
        # self.enc1 = self._block(base_channels*2, base_channels, "enc1") # 64-->32
        # =====================================
        self.enc1 = self._block(in_channels, base_channels, "enc1") # 64-->32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(base_channels, base_channels*2, "enc2") # 32-->64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(base_channels*2, base_channels*4, "enc3") # 64-->128
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(base_channels*4, base_channels*8, "enc4") # 128-->256
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(base_channels*8, base_channels*8, "bottleneck") # 256--> 256
        
        # 解码器路径
        # self.up4 = nn.Upsample(size=(125, 125), mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = self._block(base_channels*16, base_channels*4, "dec4") # 512-->128
        # self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(size=(125, 125), mode="bilinear", align_corners=True)
        self.dec3 = self._block(base_channels*8, base_channels*2, "dec3")
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = self._block(base_channels*4, base_channels, "dec2")
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self._block(base_channels*2, base_channels, "dec1")
        
        # # 输出层
        # self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        # # 初始化权重
        # self._initialize_weights()

    
    def _block(self, in_channels, out_channels, name):
        """构建基础卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
    
    # def _initialize_weights(self):
    #     """权重初始化"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        :param x: 输入张量 [B, input_channel, H, W]
                  通道0-5: 上一帧地图
                  通道6: 上一帧标签
                  通道7-12: 当前帧地图
        """
        # 预处理输入
        x = self.input_preprocess(x)
        # x = self.embed(x) # 卷积 # c=64
        
        # 编码器路径
        enc1 = self.enc1(x)  # x1 # 500*500*32
        enc2 = self.enc2(self.pool1(enc1)) # x2 # 250*250*64
        enc3 = self.enc3(self.pool2(enc2)) # x3 # 125*125*128
        enc4 = self.enc4(self.pool3(enc3)) # x4 # 62*62*256
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4)) # x5 # 31*31*256
        
        # 解码器路径 (带跳跃连接)
        dec4 = self.up4(bottleneck) # 62*62*256
        dec4 = torch.cat((dec4, enc4), dim=1) # 62*62*512
        dec4 = self.dec4(dec4) # 62*62*128
        
        dec3 = self.up3(dec4) # 125*125*128
        dec3 = torch.cat((dec3, enc3), dim=1) # 125*125*256
        dec3 = self.dec3(dec3) # 125*125*64
        
        # dec2 = self.up2(dec3)  # 250*250*64
        # dec2 = torch.cat((dec2, enc2), dim=1) # 250*250*128
        # dec2 = self.dec2(dec2) # 250*250*32
        
        # dec1 = self.up1(dec2) # 500*500*32
        # dec1 = torch.cat((dec1, enc1), dim=1) # 500*500*32
        # dec1 = self.dec1(dec1) # 500*500*16

        return dec3
    


class InputPreprocessor(nn.Module):
    """输入预处理模块 - 增强历史标签信息"""
    def __init__(self):
        super(InputPreprocessor, self).__init__()
        
        # # 历史标签特征提取
        # self.history_processor = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),
        #     nn.ELU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.ELU()
        # )

        # 历史标签特征提取
        self.history_processor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ELU()
        )
        
        # # 地图特征提取
        # self.map_processor = nn.Sequential(
        #     nn.Conv2d(12, 32, 3, padding=1),
        #     nn.ELU(),
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.ELU()
        # )

        # 地图特征提取
        self.map_processor = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.ELU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ELU(),
            # nn.Conv2d(64, 13, 1)  # 输出通道数匹配网络输入
            nn.Conv2d(64, 6, 1)  # 输出通道数匹配网络输入
        )
    
    def forward(self, x):
        # 分离输入
        prev_map = x[:, :2, :, :]     # 上一帧地图 [B, 6, H, W]
        prev_label = x[:, 2:3, :, :]   # 上一帧标签 [B, 1, H, W]
        curr_map = x[:, 3:, :, :]      # 当前帧地图 [B, 6, H, W]
        
        # 处理历史标签
        history_feat = self.history_processor(prev_label)
        
        # 处理地图数据
        maps = torch.cat([prev_map, curr_map], dim=1)
        map_feat = self.map_processor(maps)
        
        # 融合特征
        fused = torch.cat([history_feat, map_feat], dim=1)
        return self.fusion(fused)

class MultiFloorNavigator(nn.Module):
    def __init__(self, pretrained_path=None):
        super(MultiFloorNavigator, self).__init__()
        
        # 1. 共享特征提取器
        self.extractor = HistoryAwareUNet(in_channels=6)
        
        # 2. 加载 1002_actor 参数并冻结
        print(f"Loading pretrained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path)
        # 兼容处理：如果保存的是整个 state_dict
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        
        # 过滤掉不在 extractor 中的参数（如原网络的 out_conv 或 input_preprocess）
        model_dict = self.extractor.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(new_state_dict)
        self.extractor.load_state_dict(model_dict)
        

        # 冻结参数
        for param in self.extractor.parameters():
            param.requires_grad = False
        print("Pretrained parameters frozen.")



        # 3. 修改 MultiFloorNavigator 类中的 classifier_head 部分
        # 输入：[B, 192, 125, 125]
        self.classifier_head = nn.Sequential(
            # # 500x500 -> 250x250
            # nn.Conv2d(96, 32, kernel_size=3, padding=1), # stride 默认为 1
            # nn.BatchNorm2d(32),
            # nn.ELU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ELU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # # 250x250 -> 125x125 
            # nn.Conv2d(32, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ELU(inplace=True),
            # nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ELU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # 125x125*192 -> 62x62*16
            nn.Conv2d(192, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 62x62*16 -> 31x31*8
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 31x31*8 -> 15x15*4
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 展平并输出。最后的特征图尺寸是 15*15*4 = 900
            nn.Flatten(),
            nn.Linear(4 * 15 * 15, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 3),
        )

        

    def forward(self, current_state):
        down_x = current_state[:, :6, :, :]
        mid_x = current_state[:, 6:12, :, :]
        up_x = current_state[:, 12:18, :, :]


        # 分别提取三层特征
        feat_down = self.extractor(down_x) # 125*125*64
        feat_mid = self.extractor(mid_x) # 125*125*64
        feat_up = self.extractor(up_x) # 125*125*64
        
        # Channel 拼接: [B, 192, 125, 125]
        combined_feat = torch.cat([feat_down, feat_mid, feat_up], dim=1)
        
        # 预测 logits
        logits = self.classifier_head(combined_feat) # [B, 3]

        return logits

class MultiFloor_Double_Q_Net(nn.Module):
	def __init__(self, pretrained_path, device):
		super(MultiFloor_Double_Q_Net, self).__init__()

		self.Q1 = MultiFloorNavigator(pretrained_path=pretrained_path).to(device)
		self.Q2 = MultiFloorNavigator(pretrained_path=pretrained_path).to(device)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2