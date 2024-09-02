import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import csv
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import util_train as util

def prepare_dataset(opts):

	data_aug = not opts.no_data_aug

	if opts.location == 'workstation':
		raise NotImplementedError('Only possible location for this demo code is "suppl_code".')
	elif opts.location == 'powerstation':
		raise NotImplementedError('Only possible location for this demo code is "suppl_code".')
	elif opts.location == 'cluster':
		data_folder = '/mnt/cluster/datasets/Cholec80/frames_1fps/'
		op_paths = []
		for fold in ['1','2','3','4']:
			fold_path = os.path.join(data_folder,fold)
			op_paths += [os.path.join(fold_path,op) for op in os.listdir(fold_path)]
	elif opts.location == 'suppl_code':
		data_folder = '../data/frames_1fps/'
		op_paths = [os.path.join(data_folder,op) for op in os.listdir(data_folder)]

	if opts.split=='old':
		op_paths.sort(key=util.old_order)
		#print('train')
		train_set = load_data(op_paths[00:60],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[59:60],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[60:80],opts,data_aug=False,shuffle=False)
	elif opts.split=='tecno':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:40],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[40:48],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[48:80],opts,data_aug=False,shuffle=False)
	elif opts.split=='cuhk':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:10],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[10:14],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[14:21],opts,data_aug=False,shuffle=False)
	elif opts.split=='cuhknotest':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:10],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[10:14],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[14:15],opts,data_aug=False,shuffle=False)
	elif opts.split=='cuhk1007':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:10],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[10:14],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[14:21],opts,data_aug=False,shuffle=False)

	return train_set, val_set, test_set

def load_data(op_paths,opts,data_aug,shuffle):

	if shuffle:

		data = []
		for op_path in op_paths:
			ID = os.path.basename(op_path)
			if os.path.isdir(op_path):
				dataset = Cholec80(op_path, ID, opts, data_aug, opts.seq_len)
				data.append(dataset)
		data = ConcatDataset(data)
		data = DataLoader(data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
		data = [('shuffled',data)]

	else:

		data = []
		for op_path in op_paths:
			ID = os.path.basename(op_path)
			if os.path.isdir(op_path):
				#print(ID)
				dataset = Cholec80(op_path, ID, opts, data_aug, seq_len=1)
				dataloader = DataLoader(dataset, batch_size=opts.batch_size*opts.seq_len, shuffle=False, num_workers=opts.workers, collate_fn=collate_noshuffle)
				data.append((ID,dataloader))

	return data

# during carry hidden training or evaluation, shuffling in the dataloader is turned off
# i.e. the dataloader provides consecutive video frames along the batch dimension,
# so batch dim is actually the tmeporal dim and has tobe swapped
def collate_noshuffle(batch):

	# in the unshuffled case, "default_collate" loads video batches as Tx1xCxHxW and targets as Tx1
	data = torch.stack([b[0] for b in batch])
	target = torch.stack([b[1] for b in batch])
	data = data.transpose(0,1) # transpose to 1xTxCxHxW
	target = target.transpose(0,1) # transpose to 1xT

	return data, target

def prepare_image_features(net,opts,test_mode=False):

	train_set, val_set, test_set = prepare_dataset(opts)

	if test_mode:
		print('extracting  test features...')
		test_set = extract_features(test_set,net)
		return None,None,test_set
	else:
		print('extracting train features...')
		train_set = extract_features(train_set,net)
		print('extracting   val features...')
		val_set = extract_features(val_set,net)
		print('extracting  test features...')
		test_set = extract_features(test_set,net)
		return train_set, val_set, test_set

def extract_features(dataset,net):

	with torch.no_grad():
		net.eval()
		temp_dataset = []
		for ID,op in dataset:
			features, labels = [],[]
			for data, target in op:
				data = data.cuda()
				target = target
				feat = net.extract_image_features(data)
				features.append(feat.cpu())
				labels.append(target)
				#break
			features = torch.cat(features,dim=1)
			labels = torch.cat(labels,dim=1)
			temp_dataset.append((ID,[(features,labels)]))
			#break
		net.train()
		return temp_dataset

def prepare_batch(data,target):

	data, target = data.clone(), target.clone() # cloning prevents target from being altered during 2-step training
	data, target = data.cuda(), target.cuda()

	return data, target

class Cholec80(Dataset):
	def __init__(self, image_path, ID, opts, data_aug=False, seq_len=1):
		
		#print(ID)
		self.image_path = image_path
		self.seq_len = seq_len

		if opts.location == 'cluster':
			self.ext = 'png'
		else:
			self.ext = 'jpg'

		if data_aug:
			self.transform = A.Compose([
				A.SmallestMaxSize(max_size=opts.height+40),
				A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
				A.RandomCrop(height=opts.height, width=opts.width),
				A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
				A.RandomBrightnessContrast(p=0.5),
				A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
				ToTensorV2(),
			])
		else:
			self.transform = A.Compose([
				A.SmallestMaxSize(max_size=opts.height),
				A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
				ToTensorV2(),
			])

		### generate anticipation ground truth from tool presence
		if opts.task == 'anticipation':
			annotation_path = os.path.join(opts.annotation_folder,'tool_annotations',"video"+ID+"-tool.txt")
			with open(annotation_path, "r") as f:
				tool_presence = []
				reader = csv.reader(f, delimiter='\t')
				next(reader, None)
				for i,row in enumerate(reader):
					tool_presence.append([int(row[x]) for x in [2,4,5,6,7]])
				tool_presence = torch.LongTensor(tool_presence).permute(1,0)
			self.target = generate_anticipation_gt(tool_presence,opts.horizon)

		# load phase ground truth
		if opts.task == 'phase':
			"""
			1    Preparation
			2    Dividing Ligament and Peritoneum
			3    Dividing Uterine Vessels and Ligament
			4    Transecting the Vagina
			5    Specimen Removal
			6    Suturing
			7    Washing
			"""
			
			# phase_map = {
			# 	'TrocarPlacement': 0,
			# 	'Preparation': 1,
			# 	'CalotTriangleDissection': 2,
			# 	'ClippingCutting': 3,
			# 	'GallbladderDissection': 4,
			# 	'GallbladderPackaging': 5,
			# 	'CleaningCoagulation': 6,
			# 	'GallbladderRetraction': 7
			# }
			
			annotation_path = os.path.join(opts.annotation_folder,'phase_annotations',"video"+ID+"-phase.txt")
			with open(annotation_path, "r") as f:
				self.target = []
				reader = csv.reader(f, delimiter='\t')
				next(reader, None)
				for count,row in enumerate(reader):
					# if count % 25 == 0: # the gt has been given by 1 fps
					self.target.append(int(row[1])-1)
			self.target = torch.LongTensor(self.target)

	def __getitem__(self, index):
		img_seq, target_seq = [], []

		for k in range(self.seq_len):

			img, target = self.load_frame(index + k)
			img_seq.append(img)
			target_seq.append(target)
		
		img_seq, target_seq = torch.stack(img_seq), torch.stack(target_seq)
		return img_seq, target_seq

	def load_frame(self,index):
		target = self.target[index]

		file_name = os.path.join(self.image_path,'{:08d}.{}'.format(index,self.ext))
		img = cv2.imread(file_name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = self.transform(image=img)['image']

		return img, target

	def __len__(self):
		return len(self.target) - self.seq_len + 1

# generates the ground truth signal over time for a single tool and a single operation
def generate_anticipation_gt_onetool(tool_code,horizon):
	# initialize ground truth signal
	anticipation = torch.zeros_like(tool_code).type(torch.FloatTensor)
	# default ground truth value is <horizon> minutes
	# (i.e. tool will not appear within next <horizon> minutes)
	anticipation_count = horizon
	# iterate through tool-presence signal backwards
	for i in torch.arange(len(tool_code)-1,-1,-1):
		# if tool is present, then set anticipation value to 0 minutes
		if tool_code[i]:
			anticipation_count = 0
		# else increase anticipation value with each (reverse) time step but clip at <horizon> minutes
		else:
			anticipation_count = min(horizon, anticipation_count + 1/60) # video is sampled at 1fps, so 1 step = 1/60 minutes
		anticipation[i] = anticipation_count
	# normalize ground truth signal to values between 0 and 1
	anticipation = anticipation / horizon
	return anticipation

# generates the ground truth signal over time for a single operation
def generate_anticipation_gt(tools,horizon):
	return torch.stack([generate_anticipation_gt_onetool(tool_code,horizon) for tool_code in tools]).permute(1,0)