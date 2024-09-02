import argparse

str2bool = lambda arg: arg.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='task')

#------------------#
#-- ANTICIPATION --#
#------------------#
# input data
parser_anticipation = subparsers.add_parser('anticipation')
parser_anticipation.add_argument('--width', type=int, default=384)
parser_anticipation.add_argument('--height', type=int, default=216)
parser_anticipation.add_argument('--workers', type=int, default=2)
parser_anticipation.add_argument('--location', type=str, default='suppl_code')
parser_anticipation.add_argument('--annotation_folder', type=str, default='../data/')
parser_anticipation.add_argument('--num_train_sets', type=int, default=3)
parser_anticipation.add_argument('--output_folder', type=str, default='../output/checkpoints/anticipation/')
parser_anticipation.add_argument('--trial_name', type=str, default='trial')
# model
parser_anticipation.add_argument("--backbone", type=str, default='resnet50_gn', help="Type of model. Options: 'alexnet', 'resnet18', 'resnet50', 'convnext', 'nfnet', ...")
parser_anticipation.add_argument("--head", type=str, default='lstm', help="Temporal head. Options: 'lstm' and 'tcn'")
parser_anticipation.add_argument('--num_ins', type=int, default=5)
parser_anticipation.add_argument('--horizon', type=int, default=3)
# training
parser_anticipation.add_argument('--epochs', type=int, default=300)
parser_anticipation.add_argument('--split', type=str, default='tecno', help='"tecno (40/8/32)", "cuhk (32/8/40)" or "old (60/0/20 shuffled)"')
parser_anticipation.add_argument('--batch_size', type=int, default=1)
parser_anticipation.add_argument('--seq_len', type=int, default=64)
parser_anticipation.add_argument('--lr', type=float, default=1e-5)
parser_anticipation.add_argument('--weight_decay', type=float, default=1e-2)
parser_anticipation.add_argument('--loss_scale', type=float, default=1e-2)
parser_anticipation.add_argument('--resume', type=str, default=None)
parser_anticipation.add_argument('--cnn_weight_path', type=str, default='imagenet')
parser_anticipation.add_argument('--image_based', action='store_true')
parser_anticipation.add_argument('--only_temporal', action='store_true')
parser_anticipation.add_argument('--no_data_aug', action='store_true')
parser_anticipation.add_argument('--shuffle', action='store_true')
parser_anticipation.add_argument('--bn_off', action='store_true')
parser_anticipation.add_argument('--cheat', action='store_true')
parser_anticipation.add_argument('--freeze',action='store_true',help='freezes bottom 3 blocks of ResNet-50, ResNet-50-GN or ConvNeXt.')
parser_anticipation.add_argument('--sliding_window',action='store_true')
parser_anticipation.add_argument('--random_seed',action='store_true')

#-----------#
#-- PHASE --#
#-----------#
# input data
parser_phase = subparsers.add_parser('phase')
parser_phase.add_argument('--width', type=int, default=384)
parser_phase.add_argument('--height', type=int, default=216)
parser_phase.add_argument('--workers', type=int, default=2)
parser_phase.add_argument('--location', type=str, default='suppl_code')
parser_phase.add_argument('--annotation_folder', type=str, default='../data/')
parser_phase.add_argument('--num_train_sets', type=int, default=3)
parser_phase.add_argument('--output_folder', type=str, default='../output/checkpoints/phase/')
parser_phase.add_argument('--trial_name', type=str, default='trial')
# model
parser_phase.add_argument("--backbone", type=str, default='resnet50_gn', help="Type of model. Options: 'alexnet', 'resnet50', 'resnet50_gn', 'convnext', nfnet', ...")
parser_phase.add_argument("--head", type=str, default='lstm', help="Temporal head. Options: 'lstm' and 'tcn'")
parser_phase.add_argument('--num_classes', type=int, default=7)
# training
parser_phase.add_argument('--epochs', type=int, default=200)
parser_phase.add_argument('--split', type=str, default='tecno', help='"tecno (40/8/32)", "cuhk (32/8/40)" or "old (60/0/20 shuffled)"')
parser_phase.add_argument('--batch_size', type=int, default=1)
parser_phase.add_argument('--seq_len', type=int, default=64)
parser_phase.add_argument('--lr', type=float, default=1e-5)
parser_phase.add_argument('--weight_decay', type=float, default=1e-2)
parser_phase.add_argument('--resume', type=str, default=None)
parser_phase.add_argument('--cnn_weight_path', type=str, default='imagenet')
parser_phase.add_argument('--image_based', action='store_true')
parser_phase.add_argument('--only_temporal', action='store_true')
parser_phase.add_argument('--old_split', action='store_true')
parser_phase.add_argument('--no_data_aug', action='store_true')
parser_phase.add_argument('--shuffle', action='store_true')
parser_phase.add_argument('--bn_off', action='store_true')
parser_phase.add_argument('--cheat', action='store_true')
parser_phase.add_argument('--freeze',action='store_true',help='freezes bottom 3 blocks of ResNet-50, ResNet-50-GN or ConvNeXt.')
parser_phase.add_argument('--sliding_window',action='store_true')
parser_phase.add_argument('--random_seed',action='store_true')
