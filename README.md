# DACAT: Dual-stream Adaptive Clip-aware Time Modeling for Robust Online Surgical Phase Recognition


## 1. Preparation

### Step 1:

<details>
<summary>Download the Cholec80, M2CAI16, AutoLaparo</summary>

- Access can be requested [Cholec80](http://camma.u-strasbg.fr/datasets), [M2CAI16](http://camma.u-strasbg.fr/datasets), [AutoLaparo](https://autolaparo.github.io/).
- Download the videos for each datasets and extract frames at 1fps. E.g. for `video01.mp4` with ffmpeg, run:
```bash
mkdir /<PATH_TO_THIS_FOLDER>/data/frames_1fps/01/
ffmpeg -hide_banner -i /<PATH_TO_VIDEOS>/video01.mp4 -r 1 -start_number 0 /<PATH_TO_THIS_FOLDER>/data/frames_1fps/01/%08d.jpg
```
- We also prepare a shell file to extract at [here](src/video2img.sh)
- The final dataset structure should look like this:

```
Cholec80/
	data/
		frames_1fps/
			01/
				00000001.jpg
				00000002.jpg
				00000003.jpg
				00000004.jpg
				...
			02/
				...
			...
			80/
				...
		phase_annotations/
			video01-phase.txt
			video02-phase.txt
			...
			video80-phase.txt
		tool_annotations/
			video01-tool.txt
			video02-tool.txt
			...
			video80-tool.txt
	output/
	train_scripts/
	predict.sh
	train.sh
```

</details>

### Step 2: 

<details>
<summary>Download pretrained models  ConvNeXt V2-T</summary>

<!-- - download ConvNeXt-T [weights](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) and place here: `train_scripts/convnext/convnext_tiny_1k_224_ema.pth` -->
- download ConvNeXt V2-T [weights](https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt) and place here: `.../train_scripts/convnext/convnextv2_tiny_1k_224_ema.pt`

</details>

### Step 3: 
<details>
<summary>Environment Requirements</summary>


See [requirements.txt](requirements.txt).

</details>


## 2. Train

### 2.1 Train Feature Cache
```bash
source .../Cholec80/train.sh
```
Rename and save the checkpoint `checkpoint_best_acc.pth.tar` in `.../train_scripts/newly_opt_ykx/LongShortNet/long_net_convnextv2.pth.tar`

### 2.2 Train DACAT
Change the `.../Cholec80/train.sh`, set `python3 train_longshort.py` and 
```bash
source .../Cholec80/train.sh
```

## 3. Infer
Set the model path in `.../Cholec80/train.sh` and 
```bash
source .../Cholec80/train.sh
```

### Our trained checkpoints can be download in [google drive](https://drive.google.com/file/d/1L6PmReQY2w_3FAcSgtDYf8PnSjU_auVr/view?usp=drive_link).

## 4. Evaluate

### 4.1 Cholec80
Use the [Matlab file](src/matlab-eval/Main.m).

### 4.2 M2CAI16
Use the [Matlab file](src/matlab-eval/Main_m2cai.m).

### 4.3 Cholec80
Use the [Python file](src/AutoLaparo/train_scripts/newly_opt_ykx/evaluation_total.py/#L66).

## Reference
* [BNPitfalls (MIA 24)](https://gitlab.com/nct_tso_public/pitfalls_bn)
