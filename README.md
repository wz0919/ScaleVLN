# ScaleVLN
Official implementation of the **ICCV 2023** paper: 
<br>**Scaling Data Generation in Vision-and-Language Navigation**<br>
[**Zun Wang**](https://zunwang1.github.io/), [**Jialu Li**](https://jialuli-luka.github.io/), [**Yicong Hong**](http://www.yiconghong.me/), [Yi Wang](https://shepnerd.github.io/), [Qi Wu](http://www.qi-wu.me/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/), [Hao Tan](https://www.cs.unc.edu/~airsplay/), [Yu Qiao](https://scholar.google.com/citations?hl=en&user=gFtI-8QAAAAJ&view_op=list_works)<br>

[Paper & Appendices](https://arxiv.org/abs/2307.15644)

![teaser](./files/overall.jpg)

</p>

## Abstract
Recent research in language-guided visual navigation has demonstrated a significant demand for the diversity of traversable environments and the quantity of supervision for training generalizable agents. To tackle the common data scarcity issue in existing vision-and-language navigation datasets, we propose an effective paradigm for generating large-scale data for learning, which applies 1200+ photo-realistic environments from HM3D and Gibson datasets and synthesizes 4.9 million instruction-trajectory pairs using fully-accessible resources on the web. Importantly, we investigate the influence of each component in this paradigm on the agent's performance and study how to adequately apply the augmented data to pre-train and fine-tune an agent. Thanks to our large-scale dataset, the performance of an existing agent can be pushed up (+11\% absolute with regard to previous SoTA) to a significantly new best of 80\% single-run success rate on the R2R test split by simple imitation learning. The long-lasting generalization gap between navigating in seen and unseen environments is also reduced to less than 1\% (versus 8\% in the previous best method). Moreover, our paradigm also facilitates different models to achieve new state-of-the-art navigation results on CVDN, REVERIE, and R2R in continuous environments.

## Updates
- **2023/07/31**🔥: We release the ScaleVLN dataset, models and training codes for R2R.

- **2023/07/14**: ScaleVLN is accpeted by ICCV2023! 🎉

## TODOs

- [x] ScaleVLN dataset
- [x] Code, data and trained models for R2R
- [ ] Code, data and trained models for other downstream tasks
- [ ] Graph Construction Code
- [ ] Speaker Training Code

## Prerequisites

### Installation

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version instead of v0.1.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```
conda create --name vlnde python=3.9
conda activate vlnde
cd VLN-DUET
pip install -r requirements.txt
```

### R2R

1. Download the required data from [here](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/r2r_preprocess_data.zip) and unzip it to `VLN-DUET/datasets/R2R`. It should include three folders `annotations, connectivity, connectivity_mp3d`.

2. Download the CLIP and EnvEdit features from [here](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/features.zip) and unzip it to `VLN-DUET/datasets/R2R`. It should include one folder `features`.

3. (Optional) Download the trained models from [here](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/r2r_trained_models.zip) and unzip it to `VLN-DUET/datasets/R2R`. It should include one folder `trained_models`.

4. Download pretrained lxmert from [here](https://nlp.cs.unc.edu/data/model_LXRT.pth) and place it at `VLN-DUET/datasets/pretrained/LXMERT`.


## Running

### Pre-training

We use Two NVDIA A100 GPUs for pre-training agents on ScaleVLN.

```bash
bash run_r2r_b14.sh "0,1" 45008
bash run_r2r_h14.sh "0,1" 45009
```


### Fine-tuning

We use one NVDIA A100 GPU for fine-tuning our agents.

```
bash scripts/r2r_b16_mix.sh 0
bash scripts/r2r_h14_envedit_mix.sh 0
...
```

## Citation
Please cite our paper:
```
@InProceedings{wang2023scalevln,
    author    = {Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu, Mohit Bansal, Stephen Gould, Hao Tan, Yu Qiao},
    title     = {Scaling Data Generation in Vision-and-Language Navigation},
    booktitle = {ICCV 2023},
    year      = {2023}
}
```

## Acknowledgement

We thank the developers of [DUET](https://github.com/cshizhe/VLN-DUET), [EnvDrop](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-VLN), [Co-Mod GAN](https://github.com/zsyzzsoft/co-mod-gan), [Discrete-Continuous VLN](https://github.com/YicongHong/Discrete-Continuous-VLN), [HAMT](https://github.com/cshizhe/VLN-HAMT) for their public code release.
