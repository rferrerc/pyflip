#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
##SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a30:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:h100:1
##SBATCH --gres=gpu
##SBATCH --gres=gpu:a100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=2 ## how many cpus or processors do you need on each computer
#SBATCH --time=18:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=80G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=hip  ## When you run squeue -u NETID this is how you can identify the job
## cd /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/

## For NOTMEDIAN simulated data: lr 0.0000000003
## For NONMEDIAN real data: lr ~ 0.000000001
module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model

## TWA 
## CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV.py \
##     --data_dir ./4050_data_for_diff_modeling/group_2_before  \
##     --measurement_file TWA-23A/target_0_ints.npy  \
##     --reference_file REF-HD-89063/reference_7_ints.npy \
##     --scene_name REAL_SMOOTH_BOTH_PLANES_wMEDIAN2\
##     --exp_name smooth_06\
##     --iters 1000 --ref_cutoff_iter 1800\
##     --star_offset_x 0.4 --star_offset_y -0.8\
##     --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 2 --lr 0.000000003 --smooth 0.6 ##--stage_fluxpos_cutoff_iter 800 #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt


## HR8799
## Reference decent LR: 0.000000001 
## reference LR: decent too for full param is 0.0000000035 (tried 10 times this)
## reference LR: decent too for full param is 0.000000005 was good too
## lr = 0.000000004 was okay, lr = 0.000000008 not bad! maybe even better?
## Attempt one at super small fov:
##CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter.py \

## CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter_tryincnoise.py \
##         --data_dir ./HR8799stuff/before_data_newversion2_SHIFTEDLYOT  \
##         --measurement_file HR8799_original/target_006_00001.npy  \
##         --reference_file REF-HD220657_original/reference_00001.npy \
##         --scene_name HR8799_ref_only_freezepos\
##         --px_mask_file REF-HD220657_original/pixel_masks.npy \
##         --ref_which_int 0 --sci_which_int 0 \
##         --exp_name ref1_dith1_int0_constrain_2kiters_poslr1\
##         --iters 2000 --sci_targ_name HR8799 --ref_cutoff_iter 30000 \
##         --star_offset_x 0.9 --star_offset_y 0.3 --no_median --freeze_position --be_normal\
##         --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.000000004 --stage_fluxpos_cutoff_iter 200  #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd

## Group 2 was on the ~14th, group 4 was on the ~16th

##CUDA_VISIBLE_DEVICES=0 python run_JWST_LATEST_twodeltawfes_trackinjected_many_stop.py \
for ((i=0; i<=0; i++))
do
        CUDA_VISIBLE_DEVICES=0 python run_JWST_LATEST_twodeltawfes_trackinjected_many_stop_cust_input_PLUSFLATS_permirror_LATEST_FASTER.py \
                --data_dir .HIP65426/NEWSETUP  \
                --measurement_file .HIP65426/NEWGOODDATA/DATAFORREAL/HIP-65426/full_im/target_002_00001COADDALL.npy  \
                --reference_file .HIP65426/NEWGOODDATA/DATAFORREAL/HIP-68245/full_im/reference_001_00001COADDALL.npy \
                --opd_filename observation_opd.npy \
                --scene_name FASTERHIP \
                --stage_fit_flat_fields_start_iter 3500 --stage_fit_flat_fields_end_iter 5500 --fit_flat_field --even_bigger_lr_factor 7000 --fluxwindsize 30 \
                --px_mask_file .HIP65426/NEWSETUP/pixel_masks/pixel_masks2.npy \
                --ref_which_int 0 --sci_which_int 0 --blur_annulus_SNR --blur_before_signal --enhance_sci_pos_grad 10000.0 \
                --exp_name "_$i" --big_lr_factor 10.0 \
                --iters 15000 --sci_targ_name HIP65426 --ref_cutoff_iter 5500 --sci_lr_weight 0.005  --freeze_OTE_OPD_sci --forced_stop_iter 11060 \
                --star_offset_x 0.0 --star_offset_y 0.0 --no_median --freeze_position --primaryOPD_basis grid  --inject_all_in_same_frame\
                --inj_xpos 7 --inj_ypos 5 \
                --inject_how_many_random 1 --inject_how_many_sigma_flux 0 --measurement_noisemap .HIP65426/NEWGOODDATA/DATAFORREAL/HIP-65426/noise_maps/noisemap_target_002_00001COADDALL.npy --track_injected_planet\
                --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 ##--smooth 0.8 ##--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd
done




##--pxartifactsmap_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata_NEWREDUCTION/jandata/AF_Lep/maybedetectorstuff/artifacts_jan_roll2_FROMADI.npy \
##--pxartifactsmap_ref_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata_NEWREDUCTION/jandata/HD_33093/maybedetectorstuff/dither1_ref_artifacts_FROMADIWHAT.npy \
##--pxartifactsmap_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata_NEWREDUCTION/octdata/AF_Lep/maybedetectorstuff/pixelnoisy_octroll1_FROMADI.npy \
##--pxartifactsmap_ref_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata_NEWREDUCTION/octdata/HD_33093/maybedetectorstuff/pixelsnoisy_octrefdither1.npy \

##--pxartifactsmap_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata/october_obs/AF_Lep/maybedetectorstuff/pixelsnoisy_octroll1_FROMADI.npy \
##--pxartifactsmap_ref_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata/october_obs/HD_33093/maybedetectorstuff/pixelsnoisy_octrefdither1.npy \

## --data_dir /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/JAN_before  \
## --measurement_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata/january_obs/AF_Lep/full_im/target_007_00001COADDALL.npy  \
## --reference_file /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/AFLEPNEW/usabledata/january_obs/HD_33093/full_im/reference_009_00001COADDALL.npy \

## group4/TWA-47/full_im/target_035_00001.npy --track_injected_planet--stop_condition --inject_the_same_in_all_frames
## --lr 0.000000003 good for TWA! Science targets at least --insert_initial_delta_OTE_OPD --insert_initial_delta_NIRCam_OPD
## maybe try customizing when OTE OPD is tuned and when NIRCam OPD is tuned????--freeze_OTE_OPD_sci--inject_all_in_same_frame
## --freeze_OTE_OPD_sci if you want to freeze the entrance OPD!! --freeze_OTE_OPD_sci--sci_chill_end
## used 5000, 3000 before  --fit_second_wfe_offsets grid --ref_cutoff_iter 1000 --iters 2000
##        --smooth 1.2 \--OPD_loss_weight 5.0
## for ((i=0; i<=4; i++))
## GOOD VALUES: --star_offset_x 0.9 --star_offset_y 0.3 
## do
##     CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter_tryincnoise.py \
##         --data_dir ./HR8799stuff/before_data  \
##         --measurement_file HR8799_original/target_006_00001.npy  \
##         --reference_file REF-HD220657_original/reference_00001.npy \
##         --scene_name HR8799_ref_long_allints_SMOOTHonepointtwojajanew\
##         --px_mask_file REF-HD220657_original/pixel_masks.npy \
##         --ref_which_int $i --sci_which_int 0 \
##         --exp_name "ref1_dith1_int$i"\
##         --iters 3000 --sci_targ_name HR8799 --ref_cutoff_iter 20000\
##         --star_offset_x 0.4 --star_offset_y -1.2 --no_median \
##         --smooth 1.2 \
##         --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.0000000075 --stage_fluxpos_cutoff_iter 200  #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd
## done

##    --reference_noisemap REF-HD220657_original/noisemap_reference_00001.npy\
    ##     --px_mask_file REF-HD220657_original/pixel_masks.npy \