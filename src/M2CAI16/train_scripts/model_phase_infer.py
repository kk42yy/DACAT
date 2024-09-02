# @kxyang 2024.7.3


import torch
from torch import nn, optim
#from nfnets import AGC
from nfnets_optim import SGD_AGC
from networks import CNN
import util_train as util
import os

class TemporalCNN(nn.Module):

	def __init__(self,out_size,backbone,head,opts):

		super(TemporalCNN, self).__init__()

		self.cnn = CNN(out_size,backbone,opts)
		if head == 'lstm':
			self.temporal_head = LSTMHead(self.cnn.feature_size,out_size,opts.seq_len)
		self.cache = []
		self.seq_len = opts.seq_len

	
	def forward(self,x):

		x = self.extract_image_features(x) # the newest frame has been add to the deque
		x = self.temporal_head(x)

		return x

	def forward_sliding_window(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head.forward_sliding_window(x)

		return x

	def extract_image_features(self,x):

		x = x.flatten(end_dim=1)
		x = self.cnn.featureNet(x) # [1,768]
		self.cache.append(x)
		if len(self.cache) <= self.seq_len:
			pass
		else:
			self.cache.pop(0)

		return torch.vstack(self.cache).unsqueeze(0)

	def cache_reset(self, seq_len=None):
		self.cache.clear()
		if seq_len is not None:
			self.seq_len = seq_len


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
			weight = torch.Tensor([
				1.6411019141231247,
				0.19090963801041133,
				1.0,
				0.2502662616859295,
				1.9176363911137977,
				0.9840248158200853,
				2.174635818337618,
			]).cuda()
			self.criterion = nn.CrossEntropyLoss(reduction='mean',weight=weight)
			# self.criterion = nn.CrossEntropyLoss(reduction='mean')
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