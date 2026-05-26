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


