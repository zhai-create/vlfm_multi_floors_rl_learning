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
        '''
        self.up4 = nn.Upsample(size=(125, 125), mode="bilinear", align_corners=True)
        self.dec4 = self._block(base_channels*16, base_channels*4, "dec4") # 512-->128
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = self._block(base_channels*8, base_channels*2, "dec3")
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = self._block(base_channels*4, base_channels, "dec2")
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self._block(base_channels*2, base_channels, "dec1")
        '''

        # resize_gai
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = self._block(base_channels*16, base_channels*4, "dec4") # 512-->128
        self.up3 = nn.Upsample(size=(125, 125), mode="bilinear", align_corners=True)
        self.dec3 = self._block(base_channels*8, base_channels*2, "dec3")
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = self._block(base_channels*4, base_channels, "dec2")
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = self._block(base_channels*2, base_channels, "dec1")


        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        # 初始化权重
        self._initialize_weights()

    
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
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
        enc1 = self.enc1(x)  # x1
        enc2 = self.enc2(self.pool1(enc1)) # x2
        enc3 = self.enc3(self.pool2(enc2)) # x3
        enc4 = self.enc4(self.pool3(enc3)) # x4
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4)) # x5
        
        # 解码器路径 (带跳跃连接)
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # 输出层
        return self.out_conv(dec1)


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

class Double_Q_Net(nn.Module):
	def __init__(self, in_channels, device):
		super(Double_Q_Net, self).__init__()

		self.Q1 = HistoryAwareUNet(in_channels=in_channels).to(device)
		self.Q2 = HistoryAwareUNet(in_channels=in_channels).to(device)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2