# One Shot Speaker Identification
### This is the source code for the: "One Shot Speaker Identification: a Good Metric Is All You Need?" paper we wrote as part of the TAU Deep Learning 2021 course's final project.

## Data
In order to train or test our model, you will have to download the [LibriSpeech 360 hours clean dataset](https://www.openslr.org/resources/12/train-clean-360.tar.gz) for training purposes and the [validation dataset](https://www.openslr.org/resources/12/test-clean.tar.gz) and [test dataset](https://www.openslr.org/resources/12/dev-clean.tar.gz) also from the LibriSpeech project.

#### **Wav2vec weights**
We used the pretrained wav2vec network from the fairseq repository. Before you start working with our source code, make sure to download [wav2vec's weights](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt).


#### **Data Preprocessing**
First, create a new folder named - 'cut_train_data_360_full_repr'
In order to cut audio files to 'length' in seconds, run the following:

***python data_preprocessing/create_dataset.py {LibriSpeech_data_folder} {cut_train_data_360_full_repr} length***

The 'length' in the CLI command should be replaced with the amount of time you would like to cut the data (the windows size). We used a windows size of 3 seconds and recommand this configuration in order to reconstruct our paper results.
If you wish to train our model when wav2vec plays as a frozen feature extractor, you will have to run the following command:

***python data_preprocessing/create_post_wav2vec_data.py {src_cut_audio_folder} {your_target_folder} --avg***

The '--avg' flag will avarage pool the wav2vec representation. You can also choose to do maxpooling by replacing the avg with '--max' or to get a downscale by 1/15 in time to train our residual convolutional network by replacing the '--avg' flag with '--conv' flag.
You can look at examples in data_preprocessing/create_datasets_commands.


#### **Evaluation**
If you wish to evaluate the one shot speaker identification task, you will have to run the following command:
***python eval_fewshot.py***

#### Train Speaker Verification Using Metric Learning
We should be able to get the weights path using argparse and the folders of the training src and validation src.


#### Train Speaker Verification Using Classification