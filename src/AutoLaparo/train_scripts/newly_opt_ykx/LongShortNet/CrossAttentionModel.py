import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, emb_dim=768, in_channels=512, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
 
        # self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
 
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
 
        # self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, T=256, C]
        :param context: [batch_szie, T=L, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # b, c, h, w = x.shape
 
        # x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
 
        Q = self.Wq(x)  # [batch_size, T=256, emb_dim=768]
        K = self.Wk(context)  # [batch_size, seq_len=L, emb_dim]
        V = self.Wv(context)
 
        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale
 
        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]
 
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        # out = self.proj_out(out)   # [batch_size, c, h, w]
 
        # print(out.shape)
 
        return out #, att_weights
    
if __name__ == "__main__":
    ca = CrossAttention(7)
    short = torch.randn(1,64,7)
    long = torch.randn(1,256,7)
    print(ca(short, long).shape)
    # short2 = short[:,0:1,:]
    # print((ca(short, long)[:,0:1,:] == ca(short2, long)))
    # print(ca(short2, long).shape)