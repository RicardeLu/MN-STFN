import torch 
from torch import nn
from convlstm import ConvLSTMCell

    
class ConvLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding,is_elu=True):
        super(ConvLayer,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=self.kernel_size, stride= self.stride,padding=self.padding)
        self.is_elu = is_elu
        self.elu = nn.ELU()
    def forward(self,x):
        out = self.conv(x)
        if self.is_elu:
            out = self.elu(out)
        return out

class TransposeConvLayer(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding,is_elu=True,output_padding=0) -> None:
        super(TransposeConvLayer,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )
        self.elu = nn.ELU()
        self.is_elu =is_elu
    def forward(self,x):
        out = self.transpose_conv(x)
        # if self.is_bn:
        if self.is_elu:
            out = self.elu(out)
        return out


class MultConv(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,recurrent_index=5) -> None:
        super(MultConv,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.recurrent_index = recurrent_index
        self.conv_list = nn.Sequential(*[ConvLayer(input_dim,output_dim,kernel_size=kernel_size,stride=1,padding=1) for _ in range(self.recurrent_index)])
    def forward(self,x):
        out = self.conv_list(x)
        return out

class DownConv(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,is_elu) -> None:
        super(DownConv,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.conv = ConvLayer(input_dim,output_dim,kernel_size=kernel_size,is_elu=is_elu,stride=2,padding=1)
    def forward(self,x):
        out  = self.conv(x)
        return out



class MultDeConv(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,recurrent_index=5) -> None:
        super(MultDeConv,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.recurrent_index = recurrent_index
        self.deconv_list = nn.Sequential(*[TransposeConvLayer(input_dim,output_dim,kernel_size=kernel_size,stride=1,padding=1) for _ in range(self.recurrent_index) ])
    def forward(self,x):
        out  = self.deconv_list(x)
        return out

class UpDeConv(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,is_elu) -> None:
        super(UpDeConv,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.deconv = TransposeConvLayer(input_dim,output_dim,kernel_size=kernel_size,is_elu=is_elu,stride=2,padding=1,output_padding=1)
    def forward(self,x):
        out = self.deconv(x)
        return out


class SCCBlock(nn.Module):
    #start Conv ConvLSTM
    def __init__(self,input_dim,output_dim,kernel_size,is_elu = True) -> None:
        super(SCCBlock,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.conv = ConvLayer(input_dim,output_dim,1,stride=1,padding=0,is_elu=is_elu)
        self.convlstm = ConvLSTMCell(output_dim,output_dim,kernel_size,True)
    def forward(self,x,state):
        if state == None:
            state = self.convlstm.init_hidden(x.shape[0],(x.shape[2],x.shape[3]))
        x = self.conv(x)
        h, c = self.convlstm(x,state)
        return (h, c)

class CNNConvLSTMBlock(nn.Module):
    def __init__(self,in_dim,hidden_dim,kernel_size,re_index) -> None:
        super(CNNConvLSTMBlock,self).__init__()
        self.mulcnn = MultConv(hidden_dim,hidden_dim,3,re_index)
        self.downcnn = DownConv(in_dim,hidden_dim,3,True)
        self.convlstm = ConvLSTMCell(hidden_dim,hidden_dim,kernel_size,True)
    def forward(self,h,state):
        if state == None:
            state = self.convlstm.init_hidden(h.shape[0],(h.shape[2]//2,h.shape[3]//2))
        h = self.downcnn(h)
        h = self.mulcnn(h)
        h, c = self.convlstm(h,state)
        return (h, c)

class DeCNNConvLSTMBlock(nn.Module):
    def __init__(self,in_dim,hidden_dim,kernel_size,re_index) -> None:
        super(DeCNNConvLSTMBlock,self).__init__()
        self.muldecnn = MultDeConv(in_dim,in_dim,3,re_index)
        self.updecnn = UpDeConv(in_dim,hidden_dim,3,True)
        self.convlstm = ConvLSTMCell(in_dim,in_dim,kernel_size,True)
    def forward(self,h,state):
        if state == None:
            state = self.convlstm.init_hidden(h.shape[0],(h.shape[2],h.shape[3]))
        h, c = self.convlstm(h,state)
        h_n = self.muldecnn(h)
        h_n = self.updecnn(h_n)   
        return h_n, (h, c)

class FDCBlock(nn.Module):
    # Final Deconv ConvLstm
    def __init__(self,input_dim,output_dim,kernel_size,is_elu = True) -> None:
        super(FDCBlock,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.deconv = TransposeConvLayer(input_dim,output_dim,1,stride=1,padding=0,is_elu=is_elu)
        self.convlstm = ConvLSTMCell(input_dim,input_dim,kernel_size,True)
    def forward(self,h,state):
        if state == None:
            state = self.convlstm.init_hidden(h.shape[0],(h.shape[2],h.shape[3]))
        h, c = self.convlstm(h,state)
        out = self.deconv(h)
        return out,(h, c)

class NonLocAtt(nn.Module):
    def __init__(self,in_dim,hidden_dim) -> None:
        super(NonLocAtt,self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.conv_q = nn.Conv2d(in_dim,hidden_dim,1,1,bias=False)
        self.conv_kv = nn.Conv2d(in_dim,hidden_dim,1,1,bias=False)
        self.conv_o = nn.Conv2d(hidden_dim,in_dim,1,1,bias=False)
        # nn.init.xavier_normal_(self.W_tau)
    def forward(self,H,h):

        H_kv = self.conv_kv(H.reshape(H.shape[0],H.shape[1],H.shape[2],-1)) 

        h_q = self.conv_q(h)
        h_q = h_q.reshape(h_q.shape[0],h_q.shape[1],-1).permute(0,2,1)

        att = []
        for t in range(H_kv.shape[2]):
            H_t = H_kv[:,:,t,:]

            a_t = torch.matmul(h_q,H_t)

            a_t = a_t - torch.diag_embed(a_t.diagonal(dim1=1,dim2=2))
            att.append(a_t)
        att = torch.softmax(torch.concat(att,1),dim=1)


        H_kv = H_kv.reshape(H_kv.shape[0],H_kv.shape[1],-1)

        h_v = torch.matmul(H_kv,att)

        h_v = self.conv_o(h_v.reshape(h.shape[0],-1,h.shape[2],h.shape[3]))
        out = h + h_v
        return out,att





class AttFDCBlock(nn.Module):
    # Final Deconv ConvLstm
    def __init__(self,input_dim,output_dim,kernel_size,is_elu = True) -> None:
        super(AttFDCBlock,self).__init__()
        self.input_channel = input_dim
        self.output_channel = output_dim
        self.kernel_size = kernel_size
        self.deconv = TransposeConvLayer(input_dim,output_dim,1,stride=1,padding=0,is_elu=is_elu)
        self.convlstm = ConvLSTMCell(input_dim,input_dim,kernel_size,True)
        self.att = NonLocAtt(input_dim,input_dim)
    def forward(self,h,state,H): 
        if state == None:
            state = self.convlstm.init_hidden(h.shape[0],(h.shape[2],h.shape[3]))
        h, c = self.convlstm(h,state)
        out, a = self.att(H,h)
        out = self.deconv(out)
        return out,(h, c),a




 

class MNSTFN(nn.Module):
    def __init__(self,in_dim,hidden_dim,forecast_step,block_index,re_index) -> None:
        super(MNSTFN,self).__init__()
        self.ec = SCCBlock(in_dim,hidden_dim,(3,3),False)
        self.eblocks = nn.Sequential(*[CNNConvLSTMBlock(hidden_dim*(2**i),hidden_dim*(2**(i+1)),(3,3),re_index) for i in range(block_index) ])
        self.fd = AttFDCBlock(hidden_dim,in_dim,(3,3),False)
        self.fblocks = nn.Sequential(*[DeCNNConvLSTMBlock(hidden_dim*(2**(i+1)),hidden_dim*(2**i),(3,3),re_index) for i in range(block_index) ])
        self.block_index = block_index
        self.forecast_step = forecast_step
    def forward(self,X):
        states = [ None for _ in range(self.block_index + 1 )]
        H = []
        for t in range(X.shape[1]):
            x = X[:,t,:]
            h, c = self.ec(x,states[0])
            H.append(h)
            states[0] = (h, c)
            for i in range(self.block_index):
                h, c =  self.eblocks[i](h,states[i+1])
                states[i+1] = (h, c)
        x_tau = torch.zeros(h.shape,device=X.device)
        H = torch.stack(H,dim=2)
        Y = []
        A = []
        for t in range(self.forecast_step):
            h = x_tau
            for i in range(self.block_index,0,-1):
                h,states[i] = self.fblocks[i-1](h,states[i])
            y, states[0],a = self.fd(h,states[0],H)
            A.append(a)
            Y.append(y)
        return torch.stack(Y,dim=1),torch.stack(A,dim=1)
    



   