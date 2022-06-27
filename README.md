# Adversarial Detection by Sensitivity Inconsistency Features(SIFD)
This repository contains code to reproduce results from the paper:

Learning to Discriminate Adversarial Examples by Sensitivity Inconsistency in IoHT Systems

![figure3 0](https://user-images.githubusercontent.com/30210177/168559777-96107dfd-b0c6-4232-8b00-19ca3a6094a7.png)


## Datesets
There are three datasets used in our experiments. Download and place the file `train.csv` and `test.csv` of the three datasets under the directory `/data/ag_news`, `/data/dbpedia` and `/data/yahoo_answers`, respectively.

- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/)
- [SST2](https://nlp.stanford.edu/sentiment/)

## Dependencies
There are one dependency for this project. Download and put `counter-fitted-vectors.txt` to the directory `/data/`.

- [Counter fitted vectors](https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip)

## Requirements
- python 3.6.13
- numpy 1.19.2
- scikit-learn 0.24.2 
- pytorch 1.10.2
- xgboost 1.5.2

## File Description
- `\models\`:  The models architecture for CNN, LSTM.
- `\utils\`: The support functions for .
- `config.py`: Settings of datasets, models, attacks and detector.
- `run_train.py`: Train target models.
- `run_attack.py`: Attack models.
- `detector_data.py`,`train_detector.py`: Extract adersarial features and train detectors.
- `wdr_data.py`,`wdr_train.py` : Reproduce WDR detection of paper [ACL_2022.That is a suspicious Reaction, interpreting logits variation to detect NLP Adversarial attacks](https://arxiv.org/pdf/2204.04636.pdf).
- `test_detect_data.py`, `detect_test.py`: Transferability experiments.

## Experiments

1. Train and test models:\
    Train:
    ```shell
    CUDA_VISIBLE_DEVICES=1 python3 run_train.py --dataset agnews --mode train --model_type bert
    ```
    (You will get a directory named like `agnews_bert` in path `/save_models/`)\
    You could also use our pretrained models in the directory `/save_models/`.
    Test:
    ```shell
    CUDA_VISIBLE_DEVICES=1 python3 run_train.py --dataset agnews --mode test --model_type bert
    ```

2. Attack the normally trained model:
    ```shell
   CUDA_VISIBLE_DEVICES=1 python3 run_attack.py  --dataset imdb --modify_rate 0.1 --attack_num 100  --model_type bert --attack textfooler
    ```
    (We sample texts from test/dev dataset from original datasets as the attack processes are inefficient, the sampled texts are save in `/data/datasetname/fielname`)

3. Extract adversarial features

    ```shell
   CUDA_VISIBLE_DEVICES=1 python3 detector_data.py --dataset imdb --model_type bert --num 1500 --atk_path textfooler_0.1_3000 
    ```
    we get the train dataset for detector from AEs and NEs, the generated data are saved in 
     `data/detector_data`)

4. Train and Test detector 

    ```shell
    CUDA_VISIBLE_DEVICES=1 python3 train_detector.py --dataset imdb --data_num 3000 --model_type bert  --feat_dim 20 --atk_path textfooler_0.1_3000
    ```
    The trained models are save in `save_models/detector_models`, such as `/data/zhanghData/SIFD-Adversarial_Detection/save_models/detector_models/imdb_bert/textfooler_0.1/xgboost/mode3_num3000_dim20.pickle`
    (**Note that som parameters are different amongs datasets, more shell vinformation is shown in train_detector.py **)

## More details

Most experiments setting have been provided in our paper. Here we provide some more details to help reproduce our results.

+ For normal training, we set `num_epochs` to `2` on CNN models and `3` on RNN models. For adversarial training, we train 10 epochs for all models except for RNN models of Yahoo! Answers dataset with `3` epochs.


## Citation

If you find this code and data useful, please consider citing the original work by authors:

```
******
```
