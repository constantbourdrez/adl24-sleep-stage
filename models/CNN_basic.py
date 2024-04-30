import torch
import torch.nn as nn

## Classical CNN model
class SimplifiedCNN(nn.Module):
    def __init__(self, dropout_prob=0.2, channels=1):
        super(SimplifiedCNN, self).__init__()
        self.dropout_prob = dropout_prob
        self.channels = channels
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=self.dropout_prob)
        )
        
        # Linear layer
        self.fc = nn.Linear(128, 5)
    
    def forward(self, x):
        # Ajouter une dimension de canal
        x = x.unsqueeze(1)
        
        # Forward pass through convolutional block
        x = self.conv_block(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Linear layer and softmax
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        
        return x