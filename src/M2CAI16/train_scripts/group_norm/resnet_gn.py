import torch
import torchvision

class GroupNorm32(torch.nn.GroupNorm):
	def __init__(self, num_channels, num_groups=32, **kargs):
		super().__init__(num_groups, num_channels, **kargs)

def resnet18_gn(pretrained=True,**kwargs): # own imagenet pretraining (not used for final paper)
	model = torchvision.models.resnet18(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = torch.load('group_norm/ImageNet-ResNet18-GN.pth.tar')['state_dict']
		state_dict = {k.replace('module.',''): v for k,v in state_dict.items()}
		model.load_state_dict(state_dict)
	return model

def resnet34_gn(pretrained=True,**kwargs): # own imagenet pretraining (not used for final paper)
	model = torchvision.models.resnet34(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = torch.load('group_norm/ImageNet-ResNet34-GN.pth.tar')['state_dict']
		state_dict = {k.replace('module.',''): v for k,v in state_dict.items()}
		model.load_state_dict(state_dict)
	return model

def resnet50_gn(pretrained=True,**kwargs):
	model = torchvision.models.resnet50(norm_layer=GroupNorm32)
	if pretrained:
		state_dict = torch.load('group_norm/ImageNet-ResNet50-GN.pth')['state_dict']
		state_dict = {k.replace('module.',''): v for k,v in state_dict.items()}
		model.load_state_dict(state_dict)
	return model