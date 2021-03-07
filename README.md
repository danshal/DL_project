# One Shot Speaker Identification
### This repo includes the source code for the: "One Shot Speaker Identification: a Good Metric Is All You Need?" paper we wrote as part of the TAU Deep Learning 2021 course's final project.

## **Data**
In order to train or test our model, you should get the LibrSpeech dataset first.
In order to get the dataset, please run the following commands:
```bash script
mkdir dataset
cd dataset
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
```
This commands will download train, validation and test datasets that we used from LibriSpeech.

## **python packages**
Our code was written in pytorch, and we assume you have a stable version of pytorch.
Please run the following:
```bash script
pip install pytorch-metric-learning
pip install soundfile
pip uninstall faiss
pip install faiss-cpu
```

## **Wav2vec weights**
We used the pretrained wav2vec network from the fairseq repository. Before you start working with our source code, make sure to run the following command on the main directory:
```bash script
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
```


## **Data Preprocessing**
First, create a new folder named - 'cut_train_data_360'
In order to cut audio files to 'length' in seconds, run the following:
```bash script
python data_preprocessing/create_dataset.py {LibriSpeech_data_folder} {cut_train_data_360_full_repr} length
```

The 'length' in the CLI command should be replaced with the amount of time you would like to cut the data (the windows size). We used a windows size of 3 seconds and recommand this configuration in order to reconstruct our paper results.
If you wish to train our model when wav2vec plays as a frozen feature extractor, you will have to run the following command:
```bash script
python data_preprocessing/create_post_wav2vec_data.py {src_cut_audio_folder} {your_target_folder} --avg
```

The '--avg' flag will avarage pool the wav2vec representation into a single 512 features vector. You can also choose to do maxpooling by replacing the avg with '--max' or to get a downscale by 1/15 in time to train our residual convolutional network by replacing the '--avg' flag with '--conv' flag.
You can look at examples in data_preprocessing/create_datasets_commands.


## **Evaluate One Shot Speaker Identification**
If you wish to evaluate the one shot speaker identification task, you will have to run the following command:
```bash script
python eval_fewshot.py {model_name}
```
Please select which model you want in the {model_name} argument from the following options: Conv, FC, FC_TUNING, classification, classification_TUNING, class_distill.
This options are explained in the article.

## **Train Speaker Verification Using Metric Learning**
To train our on top model with frozen wav2vec run:
```bash script
python train_on_top_model.py
```
For wav2vec fine-tuning, run:
```bash script
python backbone_train.py
```
This will train wav2vec and our 2 FC model.
To fine-tune wav2vec with our residual convolutional model, run:
```bash script
python backbone_train.py
```

## **Train Speaker Verification Using Classification**
For classification training without distillation technique run:
```bash script
python train_supervised.py
```

To get boost with self-distillation run:
```bash script
python train_distillation.py
```