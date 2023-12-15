import numpy as np
import torch
import torch.nn as nn

class SeemFeatureNetwork(nn.Module):
    def __init__(self, depth:int, width:int, input_dim:int, args):
        super(SeemFeatureNetwork, self).__init__()
        self.depth = depth
        self.width = width
        self.args = args
        self.input_dim = input_dim
        
        self.create_net()
        
    def create_net(self):
        #TO DO : Discuss with Jaswanth about architecture of this network
        self.feature_net = [nn.Linear( self.input_dim ,self.width)] #First Layer
        for ind in range(self.depth-1):
            self.feature_out.append(nn.Linear(self.width,self.width))
        self.feature_out = nn.Sequential(*self.feature_out)
        
    def forward(self, pts_enc):
        return self.feature_out(pts_enc)