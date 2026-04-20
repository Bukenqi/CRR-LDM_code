\# Latent Diffusion Model for High-Resolution 3D Cloud Radar Reflectivity Reconstruction



This repository contains the code for the manuscript:



\*\*"A Latent Diffusion Model (LDM) for Reconstructing High-Resolution 3D Cloud Radar Reflectivity over the Pacific Ocean using Himawari-8/9 AHI Measurements"\*\*



\---



\## 1. Main Components



\### Model



\* `Model\_VAE.py`: Variational Autoencoder (VAE)

\* `Model\_DDIM\_UNet.py`: DDIM-based U-Net model



\### Training



\* `Train\_VAE.py`: training VAE

\* `Train\_CRR\_LDM\_IR.py`: The model trained using all channels

\* `Train\_CRR\_LDM\_Full.py`: The model trained using the long-wave infrared channel



\### Inference / Reconstruction



\* `CRR\_LDM\_Reconstruct\_3DRF.py`: reconstruct 3D radar reflectivity

\* `CRR\_LDM\_Gen\_samples.py`: generate samples using tested model



\### Data Processing



\* `Data\_preprocessing\_Himawari\_Cloudsat.py`: preprocessing Himawari-8/9 and CloudSat data



\### Utilities

\* `FUNC\_read\_data.py`

\* `FUNC\_plot\_image.py`

\* `FUNC\_plot\_3Dscene.py`

\* `FUNC\_analyse\_data.py`



\### Visualization



\* `Plot\_\*.py`: scripts for figures and evaluation metrics (SSIM, CRPS, FSS, etc.)



\---



\## 2. Requirements



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 3. Data



The datasets used in this study are publicly available:



\* Himawari-8/9 AHI: \[https://www.eorc.jaxa.jp/ptree/]

\* CloudSat: \[https://www.cloudsat.cira.colostate.edu/data-products/2b-geoprof]



Due to data size, the full datasets are not included in this repository.



\---



\## 4. How to Run



\### Example: Reconstruction



```bash

python CRR\_LDM\_Reconstruct\_3DRF.py

```



\### Example: Generate Samples



```bash

python CRR\_LDM\_Gen\_samples.py

```



\---



\## 5. Notes



\* Paths to datasets need to be configured manually in the scripts.

\* Users should modify configuration parameters according to their environment.



\---



\## 6. Citation



If you use this code, please cite:



\[A Latent Diffusion Model (LDM) for Reconstructing High-Resolution 3D Cloud Radar Reflectivity over the Pacific Ocean using Himawari-8/9 AHI Measurements]



\---
