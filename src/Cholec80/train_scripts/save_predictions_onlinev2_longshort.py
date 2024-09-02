import torch
from random import shuffle
from tqdm import tqdm
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from newly_opt_ykx.dataloader import Cholec80Test
from model_anticipation import AnticipationModel
from newly_opt_ykx.LongShortNet.model_phase_maxr_v1_maxca import PhaseModel
import util_train as util
import os
import pandas as pd

opts = parser.parse_args()

# assumes <opts.resume> has form "output/checkpoints/<task>/<trial_name>/models/<checkpoint>.pth.tar"
suffix = 'predv2_DACAT'

out_folder = os.path.dirname(os.path.dirname(opts.resume)).replace('/checkpoints/','/predictions/')
gt_folder = os.path.join(out_folder,'gt')
pred_folder = os.path.join(out_folder,suffix)
os.makedirs(gt_folder,exist_ok=True)
os.makedirs(pred_folder,exist_ok=True)

if opts.task == 'anticipation':
	model = AnticipationModel(opts,train=False)
if opts.task == 'phase':
	model = PhaseModel(opts,train=False)

if opts.only_temporal:
	_,_,test_set = prepare_image_features(model.net_short,opts,test_mode=True)
else:
	# _,_,test_set = prepare_dataset(opts)
	data_folder = '../data/frames_1fps/'
	op_paths = [os.path.join(data_folder,op) for op in os.listdir(data_folder)]
	if opts.split=='cuhk':
		op_paths.sort(key=os.path.basename)
		test_set  = []
		for op_path in op_paths[40:80]:
			ID = os.path.basename(op_path)
			if os.path.isdir(op_path):
				test_set.append((ID,op_path))

def Unitconversion(flops, params, throughout):
    print("params : {} M".format(round(params / (1000**2), 2)))
    print("flop : {} G".format(round(flops / (1000**3), 2)))
    print("throughout: {} Images/Min".format(throughout * 60))

def weight_test(model, x):
    import time
    start_time = time.time()
    _ = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout

with torch.no_grad():

	if opts.cheat:
		model.net_short.train()
	else:
		model.net_short.eval()
		model.net_long.eval()

	for ID,op_path in test_set:

		predictions = []
		labels = []

		if not opts.image_based:
			model.net_short.temporal_head.reset()
			model.net_short.CA_temp_head.reset()
		
		model.net_long.cache_reset() # reset the feature cache for new video
		model.net_short.cache_reset()
  
		model.metric_meter['test'].start_new_op()
		offline_cholec80_test = Cholec80Test(op_path, ID, opts, seq_len=1)
  
		for _ in tqdm(range(len(offline_cholec80_test))):
			
			data, target = next(offline_cholec80_test)
			data, target = prepare_batch(data,target)

			if opts.shuffle:
				model.net_short.temporal_head.reset()

			if opts.sliding_window:
				output = model.forward_sliding_window(data)
			else:
				output = model.forward(data)
			
			if isinstance(output, tuple):
				output = output[0]
			output = [output[-1][:,-1:,:]]
			target = target[:,-1:]
			
			model.update_stats(
				0,
				output,
				target,
				mode='test'
			)

			if opts.task == 'phase':
				_,pred = output[-1].max(dim=2)
				predictions.append(pred.flatten())
				labels.append(target.flatten())
			
			elif opts.task == 'anticipation':
				pred = output[-1][0]
				pred *= opts.horizon
				target *= opts.horizon
				predictions.append(pred.flatten(end_dim=-2))
				labels.append(target.flatten(end_dim=-2))

		predictions = torch.cat(predictions)
		labels = torch.cat(labels)
	
		if opts.task == 'phase':
			predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Phase'])
			labels = pd.DataFrame(labels.cpu().numpy(),columns=['Phase'])
			
		elif opts.task == 'anticipation':
			predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])
			labels = pd.DataFrame(labels.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])


		predictions.to_csv(os.path.join(pred_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		labels.to_csv(os.path.join(gt_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		print('saved predictions/labels for video {}'.format(ID))


	epoch = torch.load(opts.resume)['epoch']
	model.summary(log_file=os.path.join(pred_folder, 'log.txt'), epoch=epoch)
	from visualization.Visualize import visual_main
	visual_main(out_folder, suffixpred=suffix[4:])
