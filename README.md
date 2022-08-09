# SaCTI

Official code for the paper "". If you use this code please cite our paper.
 
## Requirements
Python 3.9.x
Pytorch 1.11.0
Cuda 11.6
Transformers(huggingface) 4.17.0
sklearn:0.22.1

Please run requirements.txt for dependecies installations.


Data
Data is provided in data folder of this repo.
List of data provided:

1.English
2.Marathi
3.saCTI-base coarse
4.saCTI-base fine
5.saCTI-large coarse
6.saCTI-large fine


How to train model
To train the model you have to run main.py file with 4 command line arguments.
Arguments are:
1.model_path: path to save model.
2.experiment: Name of experiment which you want to run. The list of experiments are given below. Default:saCTI-base coarse
3.epochs: Number of epochs. Default:70
4.batch_size: Size of batch. Default:50

Running the code
**python3 main.py --model_path='save_models' --experiment='english' --epochs=70 --batch_size=75**

Name of different experiments:
1. english <br />
2. marathi
3. saCTI-base coarse
4. saCTI-base fine
5. saCTI-large coarse
6. saCTI-large fine
(Please check data_config.py file for the same.)

## Citation

Acknowledgements
Much of the base code is from ["trankit"](https://github.com/nlp-uoregon/trankit)
















