# Deep Code Search

PyTorch implementation of [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf).

## Dependency
> Tested in MacOS 10.12, Ubuntu 16.04
* Python 3.6
* PyTorch 
* tqdm

 ```
 pip install -r requirements.txt
 ```
 

## Code Structures

 - `models`: neural network models for code/desc representation and similarity measure.
 - `modules.py`: basic modules for model construction.
 - `train.py`: train and validate code/desc representaton models; 
 - `repr_code.py`: encode code into vectors and store them to a file; 
 - `search.py`: perform code search;
 - `configs.py`: configurations for models defined in the `models` folder. 
   Each function defines the hyper-parameters for the corresponding model.
 - `data_loader.py`: A PyTorch dataset loader.
 - `utils.py`: utilities for models and training. 

## Pretrained Model

   If you want a quick test, [here](https://drive.google.com/file/d/1xpUXsSFbULYEAs8low5zQZWK7-wmqTNO/view?usp=sharing) is a pretrained model. Put it in `./output/JointEmbeder/github/202106140524/models/` and run:

   ```
   python repr_code.py --reload_from 4000000
   python search.py --reload_from 4000000
   ```
   
 
## Usage

   ### Data Preparation
  The `/data` folder provides a small dummy dataset for quick deployment.  
  To train and test our model:
  
  1) Download and unzip real dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing).
  
  2) Replace each file in the `/data` folder with the corresponding real file. 
  
   ### Configuration
   Edit hyper-parameters and settings in `config.py`

   ### Train
   
   ```bash
   python train.py --model JointEmbeder -v
   ```
   <img src="https://user-images.githubusercontent.com/6091014/125632961-36df8a55-5a1e-4d90-b96d-5abca8e90e0e.png" width=50% height=50%>
   ### Code Embedding
   
   ```bash
   python repr_code.py --model JointEmbeder -t XXX --reload_from YYY
   ```
   where `XXX` stands for the timestamp, and `YYY` represents the iteration with the best model.
   
   ### Search
   
   ```bash
   python search.py --model JointEmbeder -t XXX --_reload_from YYY
   ```
   where `XXX` stands for the timestamp, and `YYY` represents the iteration with the best model.
   
   Here is a screenshot of code search:
   
   <img src="/home/huangxin/code/python/code-search/code/deep-code-search/txt_file/search_result_top5.png" width=100% height=100%>
