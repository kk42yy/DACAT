import torch
from torch import nn, optim
from networks import CNN
from .CrossAttentionModel import CrossAttention
import util_train as util
import os

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x=None, y=None):
        return x

class TemporalCNN(nn.Module):

	def __init__(self,out_size,backbone,head,opts,do_lstm=True,cache_size=None,CA=False):

		super(TemporalCNN, self).__init__()

		self.cnn = CNN(out_size,backbone,opts)
		if head == 'lstm':
			self.temporal_head = LSTMHead(self.cnn.feature_size,out_size,opts.seq_len)
		self.cache = []
		self.seq_len = cache_size
		self.do_lstm = do_lstm
		if CA:
			self.CA = CrossAttention(emb_dim=768)
			self.CA_temp_head = LSTMHead(self.cnn.feature_size,out_size,opts.seq_len)
		else:
			self.CA = Identity()

	
	def forward(self,x,long_term=None):
		
		if self.seq_len is None:
			x = self.extract_image_features(x) # short: [1,256,768]
		else:
			x = self.extract_image_features_cache(x) # long: [1,8192,768] the newest frame has been add to the deque
		
		if not self.do_lstm: # for Long Term
			return x
		
		if long_term is not None:
			xca = self.CA(x, long_term)
			xca = self.CA_temp_head.forward_nohidden(xca)
			# xca = self.CA_temp_head.forward(xca)
   
		x = self.temporal_head(x)

		return [(x + xca) / 2]

	def forward_sliding_window(self,x):

		x = self.extract_image_features(x)
		x = self.temporal_head.forward_sliding_window(x)

		return x

	def extract_image_features_cache(self,x):

		x = x.flatten(end_dim=1)
		x = self.cnn.featureNet(x) # [1,768]
		self.cache.append(x)
		if len(self.cache) <= self.seq_len:
			pass
		else:
			self.cache.pop(0)

		return torch.vstack(self.cache).unsqueeze(0)

	def extract_image_features(self,x):

		B = x.size(0)
		S = x.size(1)

		x = x.flatten(end_dim=1)
		x = self.cnn.featureNet(x)
		x = x.view(B,S,-1)

		return x

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

		return x

	def forward_nohidden(self, x):
		return self.out_layer(self.lstm(x)[0])

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

		self.net_long = TemporalCNN(opts.num_classes,opts.backbone,opts.head,opts,do_lstm=False,cache_size=8192).cuda()
		self.net_short = TemporalCNN(opts.num_classes,opts.backbone,opts.head,opts,CA=True,
                               cache_size=None if train else opts.seq_len).cuda()

		print('loaded ImageNet weights...')
		
		long_net_pretrain_path = os.path.join(os.path.dirname(__file__), 'long_net_convnextv2.pth.tar')
		self.net_long.load_state_dict(torch.load(long_net_pretrain_path)['state_dict'])
  
		if opts.resume is not None:
			checkpoint = torch.load(opts.resume)
			self.net_short.load_state_dict(checkpoint['state_dict'])
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
			
			self.optimizer = optim.AdamW(self.net_short.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
			if opts.resume is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print('loaded optimizer settings...')
			
			checkp_for_short = torch.load(long_net_pretrain_path)['state_dict']
			short_parameter = self.net_short.state_dict()
			short_parameter.update(checkp_for_short)
			self.net_short.load_state_dict(short_parameter)
			self.net_long.requires_grad_(False)
			del checkp_for_short, short_parameter
   
	def forward(self,data):

		output = self.net_long(data)
		output = self.net_short(data, output)

		return output

	def forward_sliding_window(self,data):

		output = self.net_short.forward_sliding_window(data)

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
				'state_dict': self.net_short.state_dict(),
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

if __name__ == "__main__":
    import os
    path = os.path.abspath(__file__)
    p = './long_net_convnextv2.pth.tar'
    print(__file__)