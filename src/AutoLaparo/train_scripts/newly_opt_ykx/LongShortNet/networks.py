import torch
from torch import nn
import torchvision
import timm
import convnext
import torch.nn.functional as F
import copy

class TemporalCNN(nn.Module):

	def __init__(self,out_size,backbone,head,opts):

		super(TemporalCNN, self).__init__()

		self.cnn = CNN(out_size,backbone,opts)
		if head == 'lstm':
			self.temporal_head = LSTMHead(self.cnn.feature_size,out_size,opts.seq_len)
		elif head == 'tcn':
			self.temporal_head = MSTCNHead(self.cnn.feature_size,out_size)
		self.cache = []
	
	def forward(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head(x)

		return x

	def forward_sliding_window(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head.forward_sliding_window(x)

		return x

	def extract_image_features(self,x):

		B = x.size(0)
		S = x.size(1)

		x = x.flatten(end_dim=1)
		x = self.cnn.featureNet(x)
 
		x = x.view(B,S,-1)

		return x

class LSTMHead(nn.Module):

	def __init__(self,feature_size,out_size,train_len,lstm_size=512):

		super(LSTMHead, self).__init__()

		self.lstm = nn.LSTM(feature_size,lstm_size,batch_first=True)
		self.out_layer = nn.Linear(lstm_size,out_size)

		self.train_len = train_len

		self.hidden_state = None
		self.prev_feat = None

	def forward(self,x):

		x, hidden_state = self.lstm(x,self.hidden_state)
		x = self.out_layer(x)

		self.hidden_state = tuple(h.detach() for h in hidden_state)
		
		return [x]
	
	def forward_sliding_window(self,x):

		#print('#')
		if self.prev_feat is not None:
			x_sliding = torch.cat((self.prev_feat,x),dim=1)
		else:
			x_sliding = x

		x_sliding = torch.cat([
			x_sliding[:,i:i+self.train_len,:] for i in range(x_sliding.size(1)-self.train_len+1)
		])
		x_sliding, _ = self.lstm(x_sliding)
		x_sliding = self.out_layer(x_sliding)

		if self.prev_feat is not None:
			#_,pred = x_sliding.max(dim=-1)
			#print(pred)
			x_sliding = x_sliding[:,-1,:].unsqueeze(dim=0)
			#_,pred = x_sliding.max(dim=-1)
			#print(pred)
		else:
			first_preds = x_sliding[0,:-1,:].unsqueeze(dim=0)
			x_sliding = x_sliding[:,-1,:].unsqueeze(dim=0)
			x_sliding = torch.cat((first_preds,x_sliding),dim=1)

		self.prev_feat = x[:,1-self.train_len:,:].detach()
		return [x_sliding]

	def reset(self):

		self.hidden_state = None
		self.prev_feat = None

class MSTCNHead(nn.Module):

	def __init__(self,feature_size,out_size,stages=2,layers=9,f_maps=64,causal_conv=True):

		super(MSTCNHead, self).__init__()

		self.mstcn = MSTCN(feature_size,out_size,stages,layers,f_maps,causal_conv)

	def forward(self,x):

		x = x.transpose(1,2)
		x = self.mstcn(x)
		x = [out.transpose(1,2) for out in x]

		return x

	def reset(self):

		pass

class CNN(nn.Module):

	def __init__(self,out_size,backbone,opts):

		super(CNN, self).__init__()

		if backbone == 'alexnet':
			self.featureNet = torchvision.models.alexnet(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'vgg11':
			self.featureNet = torchvision.models.vgg11(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'vgg16':
			self.featureNet = torchvision.models.vgg16(pretrained=True)
			self.featureNet.classifier = self.featureNet.classifier[:3]
			self.feature_size = 4096
		elif backbone == 'resnet18':
			self.featureNet = torchvision.models.resnet18(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet34':
			self.featureNet = torchvision.models.resnet34(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 512
		elif backbone == 'resnet50':
			self.featureNet = torchvision.models.resnet50(pretrained=True)
			self.featureNet.fc = Identity()
			self.feature_size = 2048
			if opts.freeze:
				for param in self.featureNet.parameters():
					param.requires_grad = False
				for param in self.featureNet.layer4.parameters():
					param.requires_grad = True
		elif backbone == 'convnext':
			self.featureNet = convnext.convnext_tiny(pretrained=False)
			self.featureNet.head = Identity()
			self.feature_size = 768
			if opts.freeze:
				for i in [0,1,2]:
					for param in self.featureNet.downsample_layers[i].parameters():
						param.requires_grad = False
					for param in self.featureNet.stages[i].parameters():
						param.requires_grad = False
		elif backbone == 'convnextv2':
			self.featureNet = convnext.convnextv2_tiny(pretrained=False)
			self.featureNet.head = Identity()
			self.feature_size = 768
			if opts.freeze:
				for i in [0,1,2]:
					for param in self.featureNet.downsample_layers[i].parameters():
						param.requires_grad = False
					for param in self.featureNet.stages[i].parameters():
						param.requires_grad = False
		elif backbone == 'nfnet':
			self.featureNet = timm.create_model('nfnet_l0', pretrained=True)
			#self.featureNet.head.fc = Identity()
			self.featureNet.head.fc = nn.Linear(in_features=2304, out_features=4096, bias=True)
			self.feature_size = 4096
			# TODO: test if the FC really influences performances
			# NOTES ON NFNET: can only get acceptable results with following hyperparams: BS:24, LR:1e-4, lossX3 (possibly L2:2e-5, w/ CLS)
		self.out_layer = nn.Linear(self.feature_size,out_size)

	def forward(self,x):

		B = x.size(0)
		S = x.size(1)

		x = x.flatten(end_dim=1)
		x = self.featureNet(x)
		x = self.out_layer(x)
		x = x.view(B,S,-1)

		return [x]

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class MSTCN(nn.Module):
	def __init__(self, f_dim, out_features, stages=2, layers=9, f_maps=64, causal_conv=True):
		self.num_stages = stages
		self.num_layers = layers
		self.num_f_maps = f_maps
		self.dim = f_dim
		self.num_classes = out_features
		self.causal_conv = causal_conv
		print(
			f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
			f" {self.num_f_maps}, dim: {self.dim}")
		super(MSTCN, self).__init__()
		self.stage1 = TCN(self.num_layers,
						  self.num_f_maps,
						  self.dim,
						  self.num_classes,
						  causal_conv=self.causal_conv)
		self.stages = nn.ModuleList([
			copy.deepcopy(
				TCN(self.num_layers,
					self.num_f_maps,
					self.num_classes,
					self.num_classes,
					causal_conv=self.causal_conv))
			for s in range(self.num_stages - 1)
		])

	def forward(self, x):
		out_classes = self.stage1(x)
		outputs_classes = out_classes.unsqueeze(0)
		for s in self.stages:
			out_classes = s(F.softmax(out_classes, dim=1))
			outputs_classes = torch.cat(
				(outputs_classes, out_classes.unsqueeze(0)), dim=0)
		return outputs_classes

class TCN(nn.Module):
	def __init__(self,
				 num_layers,
				 num_f_maps,
				 dim,
				 num_classes,
				 causal_conv=True):
		super(TCN, self).__init__()
		self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

		self.layers = nn.ModuleList([
			copy.deepcopy(
				DilatedResidualLayer(2**i,
									 num_f_maps,
									 num_f_maps,
									 causal_conv=causal_conv))
			for i in range(num_layers)
		])
		self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

	def forward(self, x):
		out = self.conv_1x1(x)
		for layer in self.layers:
			out = layer(out)
		out_classes = self.conv_out_classes(out)
		return out_classes

class DilatedResidualLayer(nn.Module):
	def __init__(self,
				 dilation,
				 in_channels,
				 out_channels,
				 causal_conv=True,
				 kernel_size=3):
		super(DilatedResidualLayer, self).__init__()
		self.causal_conv = causal_conv
		self.dilation = dilation
		self.kernel_size = kernel_size
		if self.causal_conv:
			self.conv_dilated = nn.Conv1d(in_channels,
										  out_channels,
										  kernel_size,
										  padding=(dilation *
												   (kernel_size - 1)),
										  dilation=dilation)
		else:
			self.conv_dilated = nn.Conv1d(in_channels,
										  out_channels,
										  kernel_size,
										  padding=dilation,
										  dilation=dilation)
		self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
		self.dropout = nn.Dropout()

	def forward(self, x):
		out = F.relu(self.conv_dilated(x))
		if self.causal_conv:
			out = out[:, :, :-(self.dilation * 2)]
		out = self.conv_1x1(out)
		out = self.dropout(out)
		return (x + out)