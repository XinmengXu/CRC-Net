import torch
import torch.nn as nn
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info

class CrossLocal(nn.Module):
    def __init__(
        self,
        dim1,
        dropout = 0.5,
    ):
        super().__init__()

        self.l = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(),
            nn.Conv2d(dim1, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.Sigmoid())	

        self.g = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=1, stride=1, padding = 0),
            nn.BatchNorm2d(dim1))		
        self.a = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=1, stride=1, padding = 0),
            nn.BatchNorm2d(dim1))	
			
        self.softmax = nn.Softmax(dim=-1)        
        self.dropout = nn.Dropout(dropout)
        self.timepool = nn.AdaptiveAvgPool2d((1, None))
        self.frequencypool = nn.AdaptiveAvgPool2d((None, 1))


    def forward(self, x, y):

        
        y = self.g(y)
        x = self.a(x)
       
        z = x + y
	
        z = self.l(z)

        xn = x * z
        return xn

class FeedForwardNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=1,
                               stride=1)

        self.layer1 = nn.Linear(251, 500)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(251, 251)

    def forward(self, x):
        out1 = self.tanh(self.conv1(x))
        out2 = self.sigmoid(self.conv2(x))
        x = out1 * out2
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class Local(nn.Module):
    def __init__(
        self,
        dim1,
        dropout = 0.5,
    ):
        super().__init__()

        self.l = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Conv1d(dim1, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1),
            nn.Sigmoid())	

        self.g = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim1),
            nn.ReLU())		
        self.a = nn.Sequential(
            nn.Conv1d(dim1, dim1, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim1),
            nn.ReLU())			



    def forward(self, x, y):

        
        y = self.g(y)
        x = self.a(x)
       
        z = x + y
	
        z = self.l(z)

        xn = x * z
        return xn, z

class GLDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_2 = NonCausalConvBlock(128, 256)
		
        self.tran_conv_block_2 = NonCausalTransConvBlock(512, 128)
        self.ffn = FeedForwardNetwork(256) 
        self.norm = nn.LayerNorm([7, 251])
        self.fc = nn.Sequential(
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256)
        )
        self.gld = GLD(1792)
        self.softmax = nn.Softmax(dim=1)  
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))	
        self.sigmoid = nn.Sigmoid()
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg2 = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):


        x0 = self.conv_block_2(x)

        x01 = self.ffn(x0)
        x01 = self.norm(x0 + x01)		
		
        xo_1, xo_2 = self.gld(x01)


##########################################################
        x = xo_1 + xo_2
        #x = self.conv2(x)
        x_pool = self.avg1(x).squeeze(-1).squeeze(-1)
        x_11 = self.fc(x_pool).unsqueeze(-1).unsqueeze(-1)
        x_22 = self.fc(x_pool).unsqueeze(-1).unsqueeze(-1)
		
        x = torch.cat([x_11, x_22], dim = 1)
        x = self.softmax(x)
        x3 = torch.chunk(x, 2, dim = 1)

        x1 = xo_1 * x3[0]
        x2 = xo_2 * x3[1]	

        # #x_f = torch.cat([x1, x2], 1) 
 ######################################################       		 
        #xn = torch.cat([x1, x0], 1) + torch.cat([x2, x0], 1)
	
        x_f = self.tran_conv_block_2(torch.cat([x1, x0], 1)) + self.tran_conv_block_2(torch.cat([x2, x0], 1))

        return x_f, x1, x2

class GLD(nn.Module):
    def __init__(
        self,
        dim1,
        dropout = 0.5,
    ):
        super().__init__()

        self.g = nn.Sequential(
            nn.Conv1d(dim1, 64, kernel_size=1),
            nn.BatchNorm1d(64))		
        self.a = nn.Sequential(
            nn.Conv1d(dim1, 64, kernel_size=1),
            nn.BatchNorm1d(64))	
        self.b = nn.Sequential(
            nn.Conv1d(dim1, 64, kernel_size=1),
            nn.BatchNorm1d(64))					
        self.o = nn.Sequential(
            nn.Conv1d(64, dim1, kernel_size=1),
            nn.BatchNorm1d(dim1))			
        self.softmax = nn.Softmax(dim=-1)        
        self.dropout = nn.Dropout(dropout)
        self.local = Local(64)
    def forward(self, x, context = None, mask = None, context_mask = None):
 
        b, c, w, j = x.size()
        f = x.reshape(b, c * w, j)
        ###########for frequency axis###############		
        q_f1 = self.g(f)
        k_f1 = self.a(f)
        q_f1, z = self.local(q_f1, k_f1)
        v_f1 = self.b(f)
        atten_f = self.softmax(torch.matmul(v_f1, k_f1.permute(0, 2, 1))) # F F
        g1 = torch.matmul(atten_f, v_f1) # F T
        atten_f1 = self.softmax(torch.matmul(g1, q_f1.permute(0, 2, 1)))
        f_o1 = torch.matmul(atten_f1, g1)

        q_f2 = (1 - z) * q_f1
        k_f2 = self.a(f)
        v_f2 = self.b(f)
        atten_f_1 = self.softmax(torch.matmul(v_f2, k_f2.permute(0, 2, 1))) # F F
        g2 = torch.matmul(atten_f_1, v_f2) # F T
        atten_f2 = self.softmax(torch.matmul(g2, q_f2.permute(0, 2, 1)))
        f_o2 = torch.matmul(atten_f2, g2)

        xn1 = self.o(f_o1).reshape(b, c, w, j) 
        xn2 = self.o(f_o2).reshape(b, c, w, j)
        return xn1, xn2




class NonCausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x1 = self.conv1(x)
        x1 = x1[:, :, :, :-1]  # chomp size
        x1 = self.norm(x1)
        x1 = self.activation(x1)

        return x1


class NonCausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class CRC(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = NonCausalConvBlock(2, 16)
        self.conv_block_2 = NonCausalConvBlock(16, 32)
        self.conv_block_3 = NonCausalConvBlock(32, 64)
        self.conv_block_4 = NonCausalConvBlock(64, 128)
        self.conv_block_5 = NonCausalConvBlock(128, 256)
        self.sigmoid = nn.Sigmoid()

        
        self.ffn3 = FeedForwardNetwork(128) 
        self.gld3 = GLDB()
        self.norm3 = nn.LayerNorm([15, 251])
        self.avg3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.tran_conv_block_1 = NonCausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = NonCausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = NonCausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = NonCausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = NonCausalTransConvBlock(16 + 16, 1, is_last=True)
        self.cross1 = CrossLocal(256)
        self.cross2 = CrossLocal(128)
        self.cross3 = CrossLocal(64)
        self.cross4 = CrossLocal(32)
        self.cross5 = CrossLocal(16)	
        self.ln = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(),
								nn.Linear(128, 256))

        self.conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=257, out_features=63)
        self.fc2 = nn.Linear(in_features=257, out_features=63)
        self.fc3 = nn.Linear(in_features=63, out_features=257)
        self.fc4 = nn.Linear(in_features=63, out_features=257)
    def forward(self, x):
        x = x.unsqueeze(1)
        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)

        g1, q1, a1 = self.gld3(e_4)

        g2 = self.norm3(e_4 + g1)
        g2, q2, a2 = self.gld3(g2)
        

        g3 = self.norm3(g1 + g2)
        g3, q3, a3= self.gld3(g3)
        

        d_r2 = self.tran_conv_block_2(torch.cat((g2, self.cross2(g3, e_4)), 1))	
        d_r3 = self.tran_conv_block_3(torch.cat((d_2, self.cross3(d_r2, e_3)), 1))
        d_r4 = self.tran_conv_block_4(torch.cat((d_3, self.cross4(d_r3, e_2)), 1))
        d_r5 = self.tran_conv_block_5(torch.cat((d_4, self.cross5(d_r4, e_1)), 1))

        d_i2 = self.tran_conv_block_2(torch.cat((g2, self.cross2(g3, e_4)), 1))	
        d_i3 = self.tran_conv_block_3(torch.cat((d_2, self.cross3(d_i2, e_3)), 1))
        d_i4 = self.tran_conv_block_4(torch.cat((d_3, self.cross4(d_i3, e_2)), 1))
        d_i5 = self.tran_conv_block_5(torch.cat((d_4, self.cross5(d_i4, e_1)), 1))
       
      	out1 = self.fc3(self.fc1(d1_1))
        out2 = self.fc4(self.fc2(d1_2))
        out = torch.cat([out1, out2], dim=1)
        return out
