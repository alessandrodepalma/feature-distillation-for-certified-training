# Learning Better Certified Models from Empirically-Robust Teachers

Code for [Learning Better Certified Models from Empirically-Robust Teachers](https://openreview.net/pdf?id=3ccTPcDlNx).

**TL;DR**
We show that feature-space distillation from adversarially-trained teachers improves trade-offs between certified and standard accuracy
for state-of-the-art certified training algorithms such as [CC-IBP](https://openreview.net/pdf?id=mzyZ4wzKlM), [MTL-IBP](https://openreview.net/pdf?id=mzyZ4wzKlM), and [SABR](https://openreview.net/pdf?id=7oFuxtJtUMH).

If you use our code in your research, please cite:
```
@inproceedings{DePalma2026,
    title={Learning Better Certified Models from Empirically-Robust Teachers},
    author={De Palma, Alessandro},
    booktitle={Advances in Neural Information Processing Systems},
    year={2026}
}
```

## Trained Models

We provide trained models for one seed of the CC-Dist experiments from Table 1 (whose results are individually reported in Table 5), along with the corresponding teacher model. 

| Dataset      | Architecture | Target L-inf Perturbation | Model           | Download Link                                                                                                                                |
|--------------|--------------|---------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| CIFAR-10     | CNN7         | 2/255                     | CC-Dist         | [model](https://www.dropbox.com/scl/fi/pdg757nqx2sq4fs4fht2i/cifar10-2-255-ccdist.pt?rlkey=l6kz4malub8s4ewxggse7mfrh&st=r83i1ckz&dl=1)       |
| CIFAR-10     | CNN7         | 2/255                     | CC-Dist teacher | [model](https://www.dropbox.com/scl/fi/2nvd4bquhyvqaafdzxdn1/cifar10-2-255-teacher.pt?rlkey=525qu8crxuzi44momk0xt5l0r&st=fixpzkli&dl=1)      |
| CIFAR-10     | CNN7         | 8/255                     | CC-Dist         | [model](https://www.dropbox.com/scl/fi/q4v17quxhxzpm7e8f6qci/cifar10-8-255-ccdist.pt?rlkey=ozb1dpd1s2nds9c4bg46corwg&st=9odduigy&dl=1)       |
| CIFAR-10     | CNN7         | 8/255                     | CC-Dist teacher | [model](https://www.dropbox.com/scl/fi/mi5vsqhaaur1pr5aw2v4t/cifar10-8-255-teacher.pt?rlkey=cbkwazn2vko4a4uxslrl9gk2p&st=ocik58ii&dl=1)      |
| TinyImageNet | CNN7         | 1/255                     | CC-Dist         | [model](https://www.dropbox.com/scl/fi/dlavtkqhh1ylxujub78vo/tinyimagenet-1-255-ccdist.pt?rlkey=7zz6i9q6qdq087cq5qyrqv5zv&st=x4q9phk8&dl=1)  |
| TinyImageNet | CNN7         | 1/255                     | CC-Dist teacher | [model](https://www.dropbox.com/scl/fi/8p6c4k78t755rsnadzqoq/tinyimagenet-1-255-teacher.pt?rlkey=h6edusjvdai933a0qx6b4r0gc&st=zs4k7czn&dl=1) |
| ImageNet64   | CNN7         | 1/255                     | CC-Dist         | [model](https://www.dropbox.com/scl/fi/clq3pyjl23d3kdpfrh7ib/imagenet64-1-255-ccdist.pt?rlkey=ef8w6wqr44cue2anrnn2bey28&st=9bycuosi&dl=1)    |
| ImageNet64   | CNN7         | 1/255                     | CC-Dist teacher | [model](https://www.dropbox.com/scl/fi/qwb5awrego6mf9lwd573j/imagenet64-1-255-teacher.pt?rlkey=fkt3fxgdpzthzeflz1z04hnoe&st=o22sb6kk&dl=1)   |

## Installation

In order to execute the code, please follow the instructions below.

```
# Create a conda environment
conda create -n ct-dist python=3.8 'setuptools<70.0'
conda activate ct-dist

# install pytorch and torchvision: see https://pytorch.org/get-started/
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

We now provide instructions to install code dependencies that require a manual installation: 
```
# install the April 2024 auto_LiRPA release from source: see https://github.com/KaidiXu/auto_LiRPA
git clone https://github.com/KaidiXu/auto_LiRPA.git
cd auto_LiRPA
git reset --hard bfb7997115c5e66327de81b43c42b23c8a0ffb1e
pip install .
cd ..

# install the OVAL branch-and-bound framework: see https://github.com/oval-group/oval-bab
git clone https://github.com/oval-group/oval-bab.git
cd oval-bab
pip install .
cd ..

# install AutoAttack
pip install git+https://github.com/fra31/auto-attack
```

Finally, the present code can be installed as follows:
```
cd ct-dist
pip install .
```

## Downloading datasets

TinyImageNet and downscaled ImageNet require the execution of scripts before running the code, with instructions provided below.
The downscaled ImageNet script additionally assumes that the raw images (Train and Val, 64x64, npz format, which can be found under the "Download downsampled image data (32x32, 64x64)" section) 
have already been downloaded from [Image-Net.org](http://image-net.org/download-images) and decompressed (e.g., into `imagenet_folder`).
CIFAR-10 is instead automatically downloaded within the code through `torchvision`. 

```
# TinyImageNet scripts
cd data/tinyimagenet
bash tinyimagenet_download.sh
cd ..

# downscaled ImageNet scripts
cd data/ImageNet64
python imagenet_data_loader.py --folder imagenet_folder_path
cd ..
```

## Execution logs and models

Execution logs can be found in the `logs`, `verification_logs`, and `aa_logs` folders for training, verification and AutoAttack evaluation jobs, respectively.
The training log contains the path to the model (needed to use it as teacher, and to verification on it) within the `"model_dir"` entry of the `"summary"` sub-dictionary.
Note that for ease of implementation, the models to be used as teachers use a different model function.
Models are saved into the directory indicated by the `--dir` argument: we will use `model_<dataset>` in the scripts below.


## Reproducing results

We here provide commands to reproduce experiments in the paper.

### Table 1
 
We used seeds 1, 2, 3, 4 and 5 (use `--seed`) besides the default seed, for a total of 6 repetitions for CC-Dist and CC-IBP training.
For consistency, CC-IBP baselines employ the CC-Dist code with `--distillation_coeff 0.0`, but the codebase also contains the CC-IBP implementation from [(De Palma et al., 2024)](https://github.com/alessandrodepalma/expressive-losses), which is slightly faster.

#### CIFAR10 - 2/255

```
mkdir -p model_cifar
# CC-Dist teacher
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=1 --num-epochs=30 --config=config/cifar_2_255.json --model=cnn7_teacher --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --l1_coeff 1e-5 --reg-lambda 0.0 --pure_adv --test-interval 3 --lr-step batch --lr-schedule cyclic --opt SGD
# CC-Dist
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ccdist --ccibp_coeff 1e-2 --distillation_coeff 5e0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher 
# CC-IBP baseline
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ccdist --ccibp_coeff 1e-2 --distillation_coeff 0.0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# Model verification
python verify.py --config=config/cifar_2_255.json --model=cnn --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive.json --oval_bab_timeout 600 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.1
# AutoAttack evaluation
python evaluate_autoattack.py --config=config/cifar_2_255.json --model=cnn --load ./trained_model_path --test_att_n_steps 50 --test_att_step_size 0.1 --test-batch-size 512
```

#### CIFAR10 - 8/255

```
mkdir -p model_cifar
# CC-Dist teacher
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=1 --num-epochs=30 --config=config/cifar.json --model=cnn7_teacher --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --l1_coeff 5e-6 --reg-lambda 0.0 --pure_adv --test-interval 3 --lr-step batch --lr-schedule cyclic --opt SGD
# CC-Dist
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ccdist --ccibp_coeff 0.5 --distillation_coeff 5e0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher  
# CC-IBP baseline
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ccdist --ccibp_coeff 0.5 --distillation_coeff 0.0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# Model verification
python verify.py --config=config/cifar.json --model=cnn --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive.json --oval_bab_timeout 600 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.1
# AutoAttack evaluation
python evaluate_autoattack.py --config=config/cifar.json --model=cnn --load ./trained_model_path --test_att_n_steps 50 --test_att_step_size 0.1 --test-batch-size 512
```

#### TinyImageNet - 1/255

```
mkdir -p model_tinyimagenet
# CC-Dist teacher
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=1 --num-epochs=100 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn7_teacher_imagenet --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 10 --l1_coeff 5e-5 --reg-lambda 0.0 --pure_adv --lr-step batch --lr-schedule cyclic --opt SGD
# CC-Dist
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --ccdist --ccibp_coeff 5e-3 --distillation_coeff 5e0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
# CC-IBP baseline
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --ccdist --ccibp_coeff 3e-3 --distillation_coeff 0.0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
# Model verification
python verify.py --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_lessmem.json --oval_bab_timeout 600 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.1 --num-class 200 --oval_trim_logitdiff
# AutoAttack evaluation
python evaluate_autoattack.py --config=config/tinyimagenet.json --model=cnn --load ./trained_model_path --test_att_n_steps 50 --test_att_step_size 0.1 --test-batch-size 128 --num-class 200
```

#### ImageNet64 - 1/255

```
mkdir -p model_imagenet64
# CC-Dist teacher
python train.py --method=fast --dir=model_imagenet64 --scheduler_opts=start=2,length=1 --num-epochs=30 --num-class 1000 --batch-size=128 --config=config/imagenet64.json --model=cnn7_teacher_imagenet --model-params=num_class=1000,linear_size=1024 --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 3 --data_loader_workers 6 --l1_coeff 5e-6 --reg-lambda 0.0 --pure_adv --lr-step batch --lr-schedule cyclic --opt SGD 
# CC-Dist
python train.py --method=fast --config=config/imagenet64.json --dir=model_imagenet64 --model=cnn_7layer_bn_imagenet --model-params=num_class=1000,linear_size=1024 --scheduler_opts=start=2,length=20 --reg-lambda=0.2 --lr-decay-milestones=60,70 --num-epochs=80 --num-class 1000 --batch-size=128 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccdist --l1_coeff 1e-5 --ccibp_coeff 5e-3 --distillation_coeff 5e0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet --teacher-model-params=num_class=1000,linear_size=1024 --test-interval 5 --data_loader_workers 6 
# CC-IBP baseline
python train.py --method=fast --config=config/imagenet64.json --dir=model_imagenet64 --model=cnn_7layer_bn_imagenet --model-params=num_class=1000,linear_size=1024 --scheduler_opts=start=2,length=20 --reg-lambda=0.2 --lr-decay-milestones=60,70 --num-epochs=80 --num-class 1000 --batch-size=128 --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --ccdist --l1_coeff 1e-5 --ccibp_coeff 5e-3 --distillation_coeff 0.0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet --teacher-model-params=num_class=1000,linear_size=1024 --test-interval 5 --data_loader_workers 6
# Model verification. In case of memory issues, decrease lirpa_crown_batch.
python verify.py --config=config/imagenet64.json --model=cnn_7layer_bn_imagenet --model-params=num_class=1000,linear_size=1024 --ib_batch_size 2000 --oval_bab_config ./bab_configs/cnn7_naive_lessmem.json --oval_bab_timeout 600 --load ./trained_model_path --test_att_n_steps 40 --test_att_step_size 0.1 --num-class 1000 --lirpa_crown_batch 700 --oval_trim_logitdiff
# AutoAttack evaluation
python evaluate_autoattack.py --config=config/tinyimagenet.json --model=cnn --load ./trained_model_path --test_att_n_steps 50 --test_att_step_size 0.1 --test-batch-size 128 --num-class 1000 --model-params=num_class=1000,linear_size=1024
```

### Table 2 

These commands assume model folders have already been created (`model_<dataset>`). 
Verification and AutoAttack evaluation commands mirror those for CC-Dist.
We used seeds 1, 2, 3, (use `--seed`) besides the default seed, for a total of 4 repetitions.
As for Table 1, baselines employ the distillation code with `--distillation_coeff 0.0`. 
The codebase also contains MTL-IBP and SABR implementations from [(De Palma et al., 2024)](https://github.com/alessandrodepalma/expressive-losses), along with a separate IBP-R implementation (`--ibpr` instead of `--ibprdist`): these are slightly faster.
Whenever not included in the command set, the teachers are the same as those for CC-Dist.

#### SABR-Dist

```
# SABR-Dist CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --sabrdist --sabr_coeff 1e-2 --distillation_coeff 5e0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher 
# SABR baseline CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --sabrdist --sabr_coeff 1e-2 --distillation_coeff 0.0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# SABR-Dist teacher CIFAR-10 8/255 
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=1 --num-epochs=30 --config=config/cifar.json --model=cnn7_teacher --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --l1_coeff 1e-6 --reg-lambda 0.0 --pure_adv --test-interval 3 --lr-step batch --lr-schedule cyclic --opt SGD
# SABR-Dist CIFAR-10 8/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --sabrdist --sabr_coeff 0.7 --distillation_coeff 5e0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# SABR baseline CIFAR-10 8/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --sabrdist --sabr_coeff 0.7 --distillation_coeff 0.0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# SABR-Dist TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --sabrdist --sabr_coeff 5e-3 --distillation_coeff 5e0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
# SABR baseline TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --sabrdist --sabr_coeff 4e-3 --distillation_coeff 0.0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
```

#### MTL-Dist

```
# MTL-Dist CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --mtldist --mtlibp_coeff 4e-3 --distillation_coeff 5e0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher 
# MTL-IBP baseline CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --mtldist --mtlibp_coeff 4e-3 --distillation_coeff 0.0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# MTL-Dist CIFAR-10 8/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --mtldist --mtlibp_coeff 0.5 --distillation_coeff 5e0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 1e-7 --teacher_path ./teacher_path --teacher_model=cnn7_teacher 
# MTL-IBP baseline CIFAR-10 8/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=180,220 --num-epochs=260 --config=config/cifar.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --mtldist --mtlibp_coeff 0.5 --distillation_coeff 0.0 --train_att_n_steps 1 --train_att_step_size 10.0 --l1_coeff 1e-7 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# MTL-Dist TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --mtldist --mtlibp_coeff 3e-3 --distillation_coeff 5e0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet 
# MTL-IBP baseline TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --mtldist --mtlibp_coeff 2e-3 --distillation_coeff 0.0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
```

#### IBP-R-Dist

```
# IBP-R-Dist teacher CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=1 --num-epochs=30 --config=config/cifar_2_255.json --model=cnn7_teacher --train_att_n_steps 10 --train_att_step_size 0.25 --test_att_n_steps 40 --test_att_step_size 0.1 --l1_coeff 2e-5 --reg-lambda 0.0 --pure_adv --test-interval 3 --lr-step batch --lr-schedule cyclic --opt SGD
# IBP-R-Dist CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ibprdist --ibpr_coeff 3e-4 --distillation_coeff 5e0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher 
# IBP-R baseline CIFAR-10 2/255
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=2,length=80 --lr-decay-milestones=120,140 --num-epochs=160 --config=config/cifar_2_255.json --model=cnn --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --ibprdist --ibpr_coeff 2e-4 --distillation_coeff 0.0 --train_att_n_steps 8 --train_att_step_size 0.25 --l1_coeff 3e-6 --attack_eps_factor 2.1 --teacher_path ./teacher_path --teacher_model=cnn7_teacher
# IBP-R-Dist TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --ibprdist --ibpr_coeff 2e-5 --distillation_coeff 5e0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet 
# IBP-R baseline TinyImageNet
python train.py --method=fast --dir=model_tinyimagenet --scheduler_opts=start=2,length=80 --reg-lambda=0.2 --lr-decay-milestones=120,140 --num-epochs=160 --num-class 200 --batch-size=128 --config=config/tinyimagenet.ibp.json --model=cnn_7layer_bn_imagenet --train_att_n_steps 1 --train_att_step_size 10.0 --test_att_n_steps 40 --test_att_step_size 0.1 --test-interval 20 --l1_coeff 5e-5 --ibprdist --ibpr_coeff 1e-5 --distillation_coeff 0.0 --teacher_path ./teacher_path --teacher_model=cnn7_teacher_imagenet
```

### Other experiments

In order to perform verification on the teachers as per Table 3, `--end_idx 500` needs to be appended to the above verification scripts in order to stop at image 500. 
Furthermore, `cnn7_naive_lessmem.json` needs to be replaced by `cnn7_naive_lessmem_crownibs.json` on TinyImageNet and ImageNet64.

Figure 1 can be reproduced by appropriately editing the `--distillation_coeff` in the scripts.

CC-Dist0, CC-Dist-Clean CC-Dist1 can be respectively run by adding the `--ccdist0_dist_loss`, `--ccdistclean_loss` , and `--ccdist1_dist_loss` flags to the CC-Dist commands, respectively.
Analogously, the logit-space distillation scheme can be run by adding `--kl_dist_loss`.
Distillation on earlier latents for CNN7 (appendix G.7) can be performed by adding the `--distillation_earlier_latent` flag.
Clean teachers (appendix G.8) can be trained by omitting `--train_att_n_steps` in the teacher commands.
Distillation from adversarial teacher features (appendix G.8) can be done via `--distillation_from_adv_t_repr`. 

The training experiments from appendices G.4 and G.6 can be replicated by suitably modifying networks (e.g., `--model=preactresnet18_teacher` and `--teacher_model=preactresnet18_teacher` for G.4's teacher and student models: see model definitions in `./models/`) and, where applicable, hyper-parameters.
When only CROWN verification is employed, the verification command should include `--crown_ver`, which disables BaB.

## Acknowledgments
Our codebase is based on the expressive losses code [(De Palma et al., 2024)](https://github.com/alessandrodepalma/expressive-losses), which was in turn built upon the codebase by [(Shi et al., 2021)](https://github.com/shizhouxing/Fast-Certified-Robust-Training).
