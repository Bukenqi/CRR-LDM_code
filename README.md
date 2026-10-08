# Latent Diffusion Model for High-Resolution 3D Cloud Radar Reflectivity Reconstruction



This repository contains the code for the manuscript:



**"A Latent Diffusion Model (LDM) for Reconstructing High-Resolution 3D Cloud Radar Reflectivity over the Pacific Ocean using Himawari-8/9 AHI Measurements"**



---



## 1. Main Components



### Model



* `Model_VAE.py`: Variational Autoencoder (VAE)

* `Model_DDIM_UNet.py`: DDIM-based U-Net model



### Training



* `Train_VAE.py`: training VAE

* `Train_CRR_LDM_IR.py`: training CRR-LDM-IR using AHI infrared channels 08–16 

* `Train_CRR_LDM_Full.py`: training CRR-LDM-Full using all 16 AHI channels



### Inference / Reconstruction



* `CRR_LDM_Reconstruct_3DRF.py`: reconstruct 3D radar reflectivity

* `CRR_LDM_Gen_samples.py`: generate samples using tested model



### Data Processing



* `Data_preprocessing_Himawari_Cloudsat.py`: preprocessing Himawari-8/9 and CloudSat data



### Utilities

* `FUNC_read_data.py`

* `FUNC_plot_image.py`

* `FUNC_plot_3Dscene.py`

* `FUNC_analyse_data.py`



### Visualization



* `Plot_*.py`: scripts for figures and evaluation metrics (SSIM, CRPS, FSS, etc.)



---



## 2. Requirements



Install dependencies:



```bash

pip install -r requirements.txt

```



---



## 3. Data



The datasets used in this study are publicly available:



* Himawari-8/9 AHI: [https://www.eorc.jaxa.jp/ptree/]

* CloudSat: [https://www.cloudsat.cira.colostate.edu/data-products/2b-geoprof]



Due to data size, the full datasets are not included in this repository.



---



## 4. How to Run



### Example: Reconstruction



```bash

python CRR_LDM_Reconstruct_3DRF.py

```



### Example: Generate Samples



```bash

python CRR_LDM_Gen_samples.py

```



---



## 5. Notes



* Paths to datasets need to be configured manually in the scripts.

* Users should modify configuration parameters according to their environment.



---



## 6. Citation



If you use this code, please cite:



[A Latent Diffusion Model (LDM) for Reconstructing High-Resolution 3D Cloud Radar Reflectivity over the Pacific Ocean using Himawari-8/9 AHI Measurements]



---


## License

This project is licensed under the Apache License 2.0.
See the LICENSE file for details.
attribution：
Parts of the CRR-LDM implementation were adapted from the
Keras Stable Diffusion example.
