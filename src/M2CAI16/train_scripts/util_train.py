import datetime
import os
from shutil import copy2
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, jaccard_score, f1_score

def prepare_output_folders(opts):

	if opts.image_based:
		temp_head = 'imageBased'
	else:
		temp_head = opts.head
	if opts.freeze:
		depth = 'frozen'
	else:
		depth = 'e2e'
	if opts.shuffle:
		shuffle = '_shuffle'
	else:
		shuffle = ''

	trial_name_full = '{}_{}_{}Split_{}_{}_lr{}_bs{}_seq{}_{}{}'.format(
		datetime.datetime.now().strftime("%Y%m%d-%H%M"),
		opts.trial_name,
		opts.split,
		temp_head,
		opts.backbone,
		opts.lr,
		opts.batch_size,
		opts.seq_len,
		depth,
		shuffle
	)

	if opts.only_temporal:
		trial_name_full = '{}_{}_{}Split_{}_{}_2step'.format(
			datetime.datetime.now().strftime("%Y%m%d-%H%M"),
			opts.trial_name,
			opts.split,
			temp_head,
			opts.backbone
		)

	output_folder = os.path.join(opts.output_folder,trial_name_full)
	print('Output directory: ' + output_folder)
	result_folder = os.path.join(output_folder,'results')
	script_folder = os.path.join(output_folder,'scripts')
	model_folder = os.path.join(output_folder,'models')
	log_path = os.path.join(output_folder,'log.txt')

	os.makedirs(result_folder,exist_ok=True)
	os.makedirs(script_folder,exist_ok=True)
	os.makedirs(model_folder,exist_ok=True)

	for f in os.listdir():
		if '.py' in f:
			copy2(f,script_folder)

	return result_folder, model_folder, log_path

def old_order(path):

	old_order = [
		'02','04','06','12','24','29','34','37','38','39','44','58','60','61','64','66','75','78','79','80',
		'01','03','05','09','13','16','18','21','22','25','31','36','45','46','48','50','62','71','72','73',
		'10','15','17','20','32','41','42','43','47','49','51','52','53','55','56','69','70','74','76','77',
		'07','08','11','14','19','23','26','27','28','30','33','35','40','54','57','59','63','65','67','68',
	]
	basename = os.path.basename(path)
	return old_order.index(basename)


def get_start_epoch(opts):

	if opts.resume is None:
		start_epoch = 1
	else:
		start_epoch = torch.load(opts.resume)['epoch'] + 1
		print('resuming at epoch {}'.format(start_epoch))

	return start_epoch

# only defined for the training dataset (since opts.shuffle only applies to train_set)
def get_iters_per_epoch(train_set,opts):

	if opts.only_temporal:
		return len(train_set)

	n_iters = sum([len(dataloader) for _,dataloader in train_set])
	if opts.shuffle:
		n_iters = n_iters // opts.seq_len
	if opts.image_based:
		n_iters = n_iters // 8

	return n_iters

class PhaseMetricMeter:

	def __init__(self, num_classes):

		self.num_classes = num_classes
		self.reset()

	def update(self,loss,output,target):

		output, target = output.flatten(end_dim=1), target.flatten(end_dim=1)

		self.loss += loss * target.size(0)
		_, predicted = torch.max(output, dim=1)

		self.pred_per_vid[-1] = np.concatenate([self.pred_per_vid[-1], predicted.cpu().numpy()])
		self.target_per_vid[-1] = np.concatenate([self.target_per_vid[-1], target.cpu().numpy()])

	def get_scores(self):

		# mean video-wise metrics

		acc_vid = np.mean([accuracy_score(gt,pred) for gt,pred in zip(self.target_per_vid,self.pred_per_vid)])
		ba_vid = np.mean([balanced_accuracy_score(gt,pred) for gt,pred in zip(self.target_per_vid,self.pred_per_vid)])

		# frame-wise metrics

		all_predictions = np.concatenate(self.pred_per_vid)
		all_targets = np.concatenate(self.target_per_vid)

		acc_frames = accuracy_score(all_targets,all_predictions)
		p = precision_score(all_targets,all_predictions,average='macro')
		r = recall_score(all_targets,all_predictions,average='macro')
		j = jaccard_score(all_targets,all_predictions,average='macro')
		f1 = f1_score(all_targets,all_predictions,average='macro')

		loss = self.loss / len(all_targets)

		return loss, acc_frames, p, r, j, f1, ba_vid, acc_vid

	def reset(self):

		self.loss = 0

		self.pred_per_vid = []
		self.target_per_vid = []

	def start_new_op(self):

		self.pred_per_vid.append(np.array([],dtype=int))
		self.target_per_vid.append(np.array([],dtype=int))


class AnticipationMetricMeter:

	def __init__(self, horizon, num_ins):

		self.eval_metric = torch.nn.L1Loss(reduction='sum')

		self.horizon = horizon
		self.num_ins = num_ins
		
		self.loss = 0
		self.l1 = 0
		self.count = 0

		self.inMAE = 0
		self.inMAE_count = 0
		self.outMAE = 0
		self.outMAE_count = 0
		self.eMAE = 0
		self.eMAE_count = 0

	def update(self,loss,output,target):

		output, target = output.flatten(end_dim=1), target.flatten(end_dim=1)

		self.loss += loss * target.size(0)
		self.l1 += (self.eval_metric(output,target) / self.num_ins).item()
		self.count += target.size(0)

		output, target = output.clone(), target.clone()
		output *= self.horizon
		target *= self.horizon

		inside_horizon = (target < self.horizon) & (target > 0)
		outside_horizon = target == self.horizon
		early_horizon = (target < (self.horizon*.1)) & (target > 0)

		abs_error = (output-target).abs()
		zeros = torch.zeros(1).cuda()

		self.inMAE += torch.where(inside_horizon, abs_error, zeros).sum(dim=0).data
		self.inMAE_count += inside_horizon.sum(dim=0).data
		self.outMAE += torch.where(outside_horizon, abs_error, zeros).sum(dim=0).data
		self.outMAE_count += outside_horizon.sum(dim=0).data
		self.eMAE += torch.where(early_horizon, abs_error, zeros).sum(dim=0).data
		self.eMAE_count += early_horizon.sum(dim=0).data

	def get_scores(self):

		loss = self.loss/self.count
		l1 = self.l1/self.count

		inMAE = (self.inMAE/self.inMAE_count).mean().item()
		outMAE = (self.outMAE/self.outMAE_count).mean().item()
		wMAE = (inMAE+outMAE)/2
		MAE = ((self.inMAE + self.outMAE) / (self.inMAE_count + self.outMAE_count)).mean().item()
		eMAE = (self.eMAE/self.eMAE_count).mean().item()

		return loss, l1, inMAE, outMAE, wMAE, MAE, eMAE

	def reset(self):

		self.loss = 0
		self.l1 = 0
		self.count = 0

		self.inMAE = 0
		self.inMAE_count = 0
		self.outMAE = 0
		self.outMAE_count = 0
		self.eMAE = 0
		self.eMAE_count = 0

	def start_new_op(self):

		pass