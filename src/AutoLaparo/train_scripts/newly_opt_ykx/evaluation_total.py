import os
import numpy as np
from sklearn.metrics import *

def read_file(gt_pt_p):
    with open(gt_pt_p, 'r+') as f:
        gt_arr = [int(i.split('\t')[1].split('\n')[0]) for i in f.readlines()[1:]]
    return np.array(gt_arr, dtype=int)

def get_scores(target_per_vid, pred_per_vid):

    # mean video-wise metrics
    acc_scores = np.array([accuracy_score(gt,pred) for gt,pred in zip(target_per_vid,pred_per_vid)])
    print(acc_scores)
    acc_vid = np.nanmean(acc_scores)
    acc_vid_std = np.nanstd(acc_scores)
    
    ba_vid = np.nanmean([balanced_accuracy_score(gt,pred) for gt,pred in zip(target_per_vid,pred_per_vid)])

    # frame-wise metrics

    all_predictions = np.concatenate(pred_per_vid)
    all_targets = np.concatenate(target_per_vid)

    acc_frames = accuracy_score(all_targets,all_predictions)
    p = precision_score(all_targets,all_predictions,average='macro')
    p_std = np.nanstd([
        [precision_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    r = recall_score(all_targets,all_predictions,average='macro')
    r_std = np.nanstd([
        [recall_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    j = jaccard_score(all_targets,all_predictions,average='macro')
    j_std = np.nanstd([
        [jaccard_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    f1 = f1_score(all_targets,all_predictions,average='macro')
    f1_std = np.nanstd([
        [f1_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    return acc_frames, p, r, j, f1, ba_vid, acc_vid, (acc_vid_std, p_std, r_std, j_std, f1_std)

def evaluate_allmetric(gtp, predp, savep):
    gt_total, pred_total = [], []
    for f in sorted(os.listdir(gtp)):
        if f.startswith('video'):
            gt_total.append(read_file(os.path.join(gtp, f)))
            pred_total.append(read_file(os.path.join(predp, f)))
    
    acc_frame, p, r, j, f1, ba_video, acc_video, std_ = get_scores(gt_total, pred_total)
    log_message = (
        f'acc frame {acc_frame*100:1.2f}\n'
        f'prec      {p*100:1.2f} ± {std_[1]*100:1.2f}, rec       {r*100:1.2f} ± {std_[2]*100:1.2f}, jacc      {j*100:1.2f} ± {std_[3]*100:1.2f}, f1        {f1*100:1.2f} ± {std_[4]*100:1.2f}\n'
        f'acc video {acc_video*100:1.2f} ± {std_[0]*100:1.2f}, ba  video {ba_video*100:1.2f}'
    )

    print(log_message)
    with open(savep, "w+") as f:
        f.write(log_message)

if __name__ == "__main__":
    gtp = ".../gt"
    predp = ".../predv2"
    savep = predp + "/metric.txt"
    evaluate_allmetric(gtp, predp, savep)