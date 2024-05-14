# adl24-sleep-stage
Project for the course "Advanced Deep-Learning", in the 3rd year of ESPCI, 2024.

- This [notebook](./sleep-edf-data.ipynb) decribes the dataset, provides links and explanations
- The data are prepared with this [script](prepare_physionet.py)
- The prepared data can be downloaded from [this drive](https://drive.google.com/drive/folders/176qhDmYUDzQg5yrv-IPzkpKSCR3Jv083?usp=sharing).
- Look at [this notebook](./sleep-edf-npz.ipynb) to play with the raw data and create data esay to use.
- A pickle is available on the drive


To better start with, the data are pre-processed. In the directory `5-cassette` you will find the data subsampled at 20 kHZ, segmented in segments of 30s each. It could be easier to start with only the cassette part and then see how telemetry can be included in the study.

The possible tasks can be split in different subprojects: what kind of models, of pre-processig, and evaluation of the performances.
These subprojects are not mutually exclusive and they can be mixed.

## Fetch data from dvc and GCP initialization

We used dvc in order to track data and to put our trained models in the cloud so as to not have the data on git (in terms of privacy and lightness of the repo). First, an account needs to be created on Google Cloud Computing. Then initialize the gcp CLI while following these instructions: https://cloud.google.com/sdk/docs/install-sdk?hl=fr

If your account is authorized, you can now fetch the latest version of the data using:
```bash
dvc pull
```
To put data in the cloud just run:
```bash
dvc add filepath_to_data
dvc push
git add .dvc
git commit -m'added data'
git push
```

## Organization of the repo

You can find the models in either 'BESTRq_classes/models.py' or 'models'. The trained models are saved in the models folder. The folder BESTRq_classes contains three python files to make the pretraining of the encoder for the BEST-RQ approach. All the training of this approach is on the 'random_projection_pretraining.ipynb' notebook.

We also made a python file 'compute_fft.py' which aims at computing and ploting the spectrograms along several channels as well as handling the random masking of this spectrograms.
