import torch
import torch.nn as nn


class SimplifiedCNN_3D(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(SimplifiedCNN_3D, self).__init__()
        self.dropout_prob = dropout_prob
        
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 9, 9), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=1),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 9, 9), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=1),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 9, 9), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=1),
            nn.Dropout(p=self.dropout_prob)
        )
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Forward pass through convolutional block
        x = self.conv_block(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply softmax
        x = self.softmax(x)
        
        return x
