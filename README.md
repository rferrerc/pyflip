## FLIP: (Ph)ysics-based Learning for Imaging Planets

Software submission for paper manuscript. 

# System requirements

1. Software dependencies used and where the code was tested on (with version numbers):
- astropy (v6.1.2)
- imageio (v2.35.1)
- matplotlib (v3.9.2)
- numpy (v2.1.0)
- pyklip (v2.8)
- scipy (v1.14.1)
- torch (v2.4.0+cu118)
- torchvision (v0.19.0+cu118)
- tqdm (v4.66.5)

2. Operating system used and where the code was tested on (with version number):
- Linux Red Hat (release 8.10)

3. Non-standard hardware used and where the code was tested on:
- NVIDIA A30 GPU

# Instalation guide

1. Instructions 
- pip install all software dependencies outlined in System requirements.
- This can be done with the provided requirements.txt file (including the exact software versions used in this work) with the command `pip install -r requirements.txt` in the terminal.

2. Install time on a normal computer
- Under 10 minutes.

# Demo 

1. Instructions to run on data
- Run the following bash command inside a virtual environment with all the dependencies outlined above and with access to an NVIDIA A30 GPU (or equivalent):

```
./scripts/run_JWST_test_HIP65426.sh
```

This will run the file `run_FLIP.py`, which contains the code to run the model. The bash file contains the filenames for the demo data included in this folder and the hyperparameters used for the HIP 65426 experiment in the paper submission.

2. Expected output
- A series of images of different points of the optimization process of both the reference star and the science target
- Two numpy arrays with the final result for the reference star calibration (stage 1 optimization) and the science target (stage 2).
- An image with a visualization of the injected planet, the science measurement, the reference measurement, the initial uncalibrated PSF model and the initial PSF subtraction
- A folder named ITERATIONS with a visualization of the injected planet every 300 iterations during stage 2, and four numpy arrays tracking the injected planet's recovered flux, the signal loss relative to the true injected flux, the SNR, and the iteration number.

3. Expected runtime demo on an NVIDIA A30 GPU:
- Around 35 minutes (running on CPU will be much slower). 

# Instructions for use

1. How to run the software on your data
- The demo instructions above will reproduce the result shown in our paper for the star HIP 65426 in Figure 2.