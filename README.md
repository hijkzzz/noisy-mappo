# Noisy-MAPPO
Codes for [Policy Perturbation via Noisy Advantage Values for Cooperative Multi-agent Actor-Critic methods](https://arxiv.org/abs/2106.14334). This repository is heavily based on https://github.com/marlbenchmark/on-policy. 

## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)

## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

All core code is located within the onpolicy folder. The algorithms/ subfolder contains code
for MAPPO. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 

## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n marl python==3.7
conda activate marl
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 Install StarCraftII [4.10](https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

``` Bash
cd ~
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip
rm -rf SC2.4.10.zip
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

* download [SMAC Maps](https://github.com/oxwhirl/smac/releases/download/v1/SMAC_Maps_V1.tar.gz), and move it to `~/StarCraftII/Maps/`.
```
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv ./SMAC_Maps ~/StarCraftII/Maps/
```

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

## 3.Train
**Please modify the hyperparameters in the shell scripts according to the Appendix of the paper.**

**Noisy-Value MAPPO (NV-MAPPO)**

```
./train_smac_value.sh 3s5z_vs_3s6z 3
```

**Noisy-Advantage MAPPO (NA-MAPPO)**

```
./train_smac_adv.sh 3s5z_vs_3s6z 3
```

**Noisy-Value IPPO (NV-IPPO)**

```
./train_smac_value_ippo.sh 3s5z_vs_3s6z 3
```

**Vanilla MAPPO (MAPPO)**

```
./train_smac_vanilla.sh 3s5z_vs_3s6z 3
```

Local results are stored in subfold scripts/results. Note that we use Tensorboard as the default visualization platform;

## Citation
```
@article{hu2021policy,
      title={Policy Perturbation via Noisy Advantage Values for Cooperative Multi-agent Actor-Critic methods}, 
      author={Jian Hu and Siyue Hu and Shih-wei Liao},
      year={2021},
      eprint={2106.14334},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```
