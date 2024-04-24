import torch
import torch.nn as nn

class SimplifiedCNN(nn.Module):
    def __init__(self, num_codebooks):
        super(self).__init__()
        
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14, 128)
        self.fc2 = nn.Linear(128, num_codebooks)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Forward pass through convolutional block
        x = self.conv_block(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        # Apply softmax
        x = self.softmax(x)
        
        return x