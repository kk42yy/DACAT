import torch
from torch import nn, optim
#from nfnets import AGC
from nfnets_optim import SGD_AGC
from networks import CNN, TemporalCNN
import util_train as util
import os

class PhaseModel(nn.Module):

	def __init__(self,opts,train=True):
		super().__init__()
		self.opts = opts
		self.train = train

		if opts.image_based:
			self.net = CNN(opts.num_classes,opts.backbone,opts).cuda()
			for param in self.net.parameters():
				param.requires_grad = True
		else:
			self.net = TemporalCNN(opts.num_classes,opts.backbone,opts.head,opts).cuda()
		#print(self.net)

		if opts.only_temporal:
			for param in self.net.cnn.parameters():
				param.requires_grad = False

		if not opts.image_based:
			if opts.cnn_weight_path != 'imagenet':
				checkpoint = torch.load(opts.cnn_weight_path)
				self.net.cnn.load_state_dict(checkpoint['state_dict'])
				print('loaded pretrained CNN weights...')
			else:
				print('loaded ImageNet weights...')

		if opts.resume is not None:
			checkpoint = torch.load(opts.resume)
			self.net.load_state_dict(checkpoint['state_dict'])
			print('loaded model weights...')

		self.metric_meter = {
			'train': util.PhaseMetricMeter(opts.num_classes),
			'val': util.PhaseMetricMeter(opts.num_classes),
			'test': util.PhaseMetricMeter(opts.num_classes)
		}

		if self.train:
			self.result_folder, self.model_folder, self.log_path = util.prepare_output_folders(opts)
			self.best_acc = 0
			self.best_f1 = 0
			# weight = torch.Tensor([
			# 	1.6411019141231247,
			# 	0.19090963801041133,
			# 	1.0,
			# 	0.2502662616859295,
			# 	1.9176363911137977,
			# 	0.9840248158200853,
			# 	2.174635818337618,
			# ]).cuda()
			# self.criterion = nn.CrossEntropyLoss(reduction='mean',weight=weight)
			self.criterion = nn.CrossEntropyLoss(reduction='mean')
			if opts.backbone == 'nfnet':
				self.optimizer = SGD_AGC(
					named_params=self.net.named_parameters(),
					lr=opts.lr,
					momentum=0.9,
					clipping=0.1,
					weight_decay=opts.weight_decay,
					nesterov=True,
				)
			else:
				self.optimizer = optim.AdamW(self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
			if opts.resume is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print('loaded optimizer settings...')
			# doesn't seem to work:
			#if opts.backbone == 'nfnet':
			#	self.optimizer = AGC(self.net.parameters(), self.optimizer, model=self.net, ignore_agc=['out_layer'])

	def forward(self,data):

		if self.opts.only_temporal:
			output = self.net.temporal_head(data)
		else:
			output = self.net(data)

		return output

	def forward_sliding_window(self,data):

		output = self.net.forward_sliding_window(data)

		return output
		
	def compute_loss_single_prediction(self,output,target):

		output = output.transpose(1,2)
		return self.criterion(output,target)

	def compute_loss(self,output,target):

		loss = [self.compute_loss_single_prediction(out,target) for out in output]
		loss = sum(loss) / len(loss)

		return loss
		
	def update_weights(self,loss):

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def reset_stats(self):

		self.metric_meter['train'].reset()
		self.metric_meter['val'].reset()
		self.metric_meter['test'].reset()
		

	def update_stats(self,loss,output,target,mode):

		output = output[-1].detach()
		target = target.detach()

		self.metric_meter[mode].update(loss,output,target)

	def summary(self,log_file=None,epoch=None):

		if self.train:

			loss_train, acc_train, _, _, _, _, _, _ = self.metric_meter['train'].get_scores()
			_, _, _, _, _, f1_val, ba_val, acc_val = self.metric_meter['val'].get_scores()
			_, _, p_test, r_test, j_test, f1_test, ba_test, acc_test = self.metric_meter['test'].get_scores()

			log_message = (
				f'Epoch {epoch:3d}: '
				f'Train (loss {loss_train:1.3f}, acc {acc_train:1.3f}) '
				f'Val (f1 {f1_val:1.3f}, ba {ba_val:1.3f}, acc {acc_val:1.3f}) '
				f'Test (Frame scores: p {p_test:1.3f}, r {r_test:1.3f}, j {j_test:1.3f}, f1 {f1_test:1.3f}; Video scores: ba {ba_test:1.3f}, acc {acc_test:1.3f}) '
			)

			checkpoint = {
				'epoch': epoch,
				'state_dict': self.net.state_dict(),
				'optimizer' : self.optimizer.state_dict(),
				'predictions': self.metric_meter['test'].pred_per_vid,
				'targets': self.metric_meter['test'].target_per_vid,
				'scores': {
					'acc': acc_test,
					'ba': ba_test,
					'f1': f1_test
				}
			}
			if self.opts.image_based:
				model_file_path = os.path.join(self.model_folder,'checkpoint_{:03d}.pth.tar'.format(epoch))
			else:
				model_file_path = os.path.join(self.model_folder,'checkpoint_current.pth.tar')
			torch.save(checkpoint, model_file_path)

			if f1_val > self.best_f1:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_f1.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_f1 = f1_val

			if acc_val > self.best_acc:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_acc.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_acc = acc_val

			print(log_message)
			log_file.write(log_message + '\n')
			log_file.flush()

		else:
			loss, aver_acc, p_test, r_test, j_test, f1_test, ba_test, acc_test = self.metric_meter['test'].get_scores()

			log_message = (
				f'Epoch {epoch:3d}: \n'
				f'Test (loss {loss:1.3f}, acc frame {aver_acc:1.3f}\n'
				f'      ba   {ba_test:1.3f}, f1   {f1_test:1.3f}\n'
				f'      acc  {acc_test:1.3f}, prec {p_test:1.3f}, rec  {r_test:1.3f}, jacc {j_test:1.3f})'
			)

			print(log_message)
			with open(log_file, "w+") as f:
				f.write(log_message)