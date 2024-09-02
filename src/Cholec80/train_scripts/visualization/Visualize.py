import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

osp = os.path.join

color_map = {
    0: {
        'name': 'Preparation',
        'color': (0, 114, 178)
    },
    1: {
        'name': 'CalotTriangleDissection', 
        'color': (230, 159, 0)
    }, 
    2: {
        'name': 'ClippingCutting', 
        'color': (0, 158, 115)
    }, 
    3: {
        'name': 'GallbladderDissection', 
        'color': (213, 94, 0)
    }, 
    4: {
        'name': 'GallbladderPackaging', 
        'color': (204, 121, 167)
    }, 
    5: {
        'name': 'CleaningCoagulation', 
        'color': (117, 112, 179)
    }, 
    6: {
        'name': 'GallbladderRetraction', 
        'color': (240, 228, 66)
    }
}

def visual_each(pred_pt_p, gt_pt_p, bmp_pt_p, ptname):
    with open(pred_pt_p, 'r+') as f:
        pred_arr = [int(i.split('\t')[1].split('\n')[0]) for i in f.readlines()[1:]]
    
    with open(gt_pt_p, 'r+') as f:
        gt_arr = [int(i.split('\t')[1].split('\n')[0]) for i in f.readlines()[1:]]
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 5))
    
    pred_colors = np.array([[color_map[i]['color'] for i in pred_arr] for _ in range(100)], dtype=np.uint8)
    gt_colors = np.array([[color_map[i]['color'] for i in gt_arr] for _ in range(100)], dtype=np.uint8)
    
    ax[0].imshow(gt_colors)
    ax[0].set_title('Ground Truth')
    ax[0].axis('off')
    
    ax[1].imshow(pred_colors)
    ax[1].set_title('Prediction')
    ax[1].axis('off')
    
    plt.tight_layout()
    fig.suptitle(ptname.split('.txt')[0], fontsize=16)
    
    plt.savefig(bmp_pt_p, bbox_inches='tight')
    plt.close()
    print(ptname)
        
def visual_main(output_folder, suffixpred='', suffixgt=''):
    pred_p, gt_p, visual_p = osp(output_folder, 'pred'+suffixpred), osp(output_folder, 'gt'+suffixgt), osp(output_folder, 'visual'+suffixpred)
    os.makedirs(visual_p, exist_ok=True)
    for ptname in sorted([i for i in os.listdir(pred_p) if i.startswith('video') and 'phase' in i]):
        pred_pt_p = osp(pred_p, ptname)
        gt_pt_p = osp(gt_p, ptname)
        bmp_pt_p = osp(visual_p, ptname.replace('.txt', '.png'))
        visual_each(pred_pt_p, gt_pt_p, bmp_pt_p, ptname)
       
    