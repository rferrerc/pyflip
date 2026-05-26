#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
##SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a30:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:h100:1
##SBATCH --gres=gpu:a100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=2 ## how many cpus or processors do you need on each computer
#SBATCH --time=18:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=80G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=rxj  ## When you run squeue -u NETID this is how you can identify the job
## cd /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/

## For NOTMEDIAN simulated data: lr 0.0000000003
## For NONMEDIAN real data: lr ~ 0.000000001
module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model


CUDA_VISIBLE_DEVICES=0 python run_FLIP.py \
        --data_dir HIP65426/model_setup  \
        --measurement_file HIP65426/demo_data/HIP-65426/full_im/target_002_00001COADDALL.npy  \
        --reference_file HIP65426/demo_data/HIP-68245/full_im/reference_001_00001COADDALL.npy \
        --opd_filename observation_opd.npy \
        --scene_name DEMO2_30newSNR2final \
        --stage_fit_flat_fields_start_iter 3500 \
        --stage_fit_flat_fields_end_iter 5500 \
        --px_mask_file HIP65426/model_setup/pixel_masks/pixel_masks.npy \
        --ref_which_int 0 --sci_which_int 0 \
        --exp_name "HIP65426test" \
        --iters 15000 --sci_targ_name HIP65426 \
        --ref_cutoff_iter 5500 --sci_lr_weight 0.005  \
        --forced_stop_iter 11000 \
        --inj_xpos 17 --inj_ypos 5 --inject_how_many_sigma_flux 30 --track_injected_planet\
        --measurement_noisemap HIP65426/demo_data/HIP-65426/noise_maps/noisemap_target_002_00001COADDALL.npy \
        --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 


