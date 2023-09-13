<p align="center" width="100%">
<img src="assert/SeismicPretrainedModel.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://github.com/shenghanlin/' target='_blank'>Hanlin Sheng
    <sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xinming  Wu<sup>1,†,‡</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xu Si<sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Jintao Li<sup>1</sup></a>&emsp;
    </br>
    <a href='https://www.huawei.com/cn/' 
    target='_blank'>Sibo Zhang <sup>2</sup></a>&emsp;
    <a href='https://www.huawei.com/cn/' 
    target='_blank'>Xudong Duan <sup>2</sup></a>&emsp;
</div>
<div>

<div align="center">
    <sup>1</sup>
    University of Science and Technology of China&emsp;
    <sup>2</sup>
    Huawei&emsp;
    </br>
    <!-- <sup>*</sup> Equal Contribution&emsp; -->
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

-----------------

# 🌟 Seismic Foundation Model (SFM)

 As shown in this workflow figure, we test the Seismic Foundation Model's performance in segmentation tasks and regression tasks, specifically in classification (i.e. seismic facies), segmentaion (i.e. seismic geobody), signal processing (i.e. denoising), inversion (i.e. reflectivity estimation), and interpolation.

This is a PyTorch/GPU implementation of the paper [Seismic Foundation Model](https://arxiv.org/abs/2309.02791):
```
@article{sheng2023seismic,
  title={Seismic Foundation Model (SFM): a new generation deep learning model in geophysics},
  author={Sheng, Hanlin and Wu, Xinming and Si, Xu and Li, Jintao and Zhang, Sibio and Duan, Xudong},
  journal={arXiv preprint arXiv:2309.02791},
  year={2023}
}
```

## 🌟 News
* **2023.9.7:** Paper is released at arxiv, and code will be gradually released.  ⌛⌛⌛
* **2023.8.7:** Github Repository Initialization (copy from Meta-Transformer). 

## &#x1F449; Pre-train & Fine-tune Code

* The pre-training instruction is in [PRETRAIN.md](SFM-Pretrain/README.md).

* The Fine-tuning instruction is in [FINETUNE.md](SFM-Finetune/README.md).


## :rocket: Model Zoo & Data Release

<!-- <details> -->
<summary> Open-source Pretrained Models </summary>
<br>
<div>

|    Model   |      Pretraining Size      |  Download |
|------------|:--------------------------:|:----------:|
| SFM-Base   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ef1xhsxytZRNjYiXJJGQAJEBZFmUiUhUuJxOyhILG88NRg?e=gGxUIb' target='_blank'>ckpt]    |  
| SFM-Base-512   |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/ES-iLYELZq1IiOor3LgDGPsBbRIXdt2wuyXeJfK-8FhM9w?e=5eURf2' target='_blank'>ckpt]    |  
| SFM-Large  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ec7ZqiAwdvtGpN5vasPnalwBXQe2qUPM_t9kdSjdkQeNIg?e=BmFlKU' target='_blank'>ckpt]    |
| SFM-Large-512  |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EXwVJmoAmRJFoenCvsIsolQBnyheFdjbejgryRj9esL2HA?e=gGsJaZ' target='_blank'>ckpt]    |    

<summary> Open-source Training & DownStream Fine-tune Task Data</summary>
<br>
<div>

|    Task   |      Size      |  Download |
|:------------------:|:--------------------------:|:----------:|
| PreTrain   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/Et8WP_voHfNMvx_kpR_iFVwBRpH3TgHsKPicCeRhXULn0g?e=f2cT2S' target='_blank'>DatFile]    |  
| Seismic Facies Classification   |         768 × 768          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/ElUKdIW6VhZOrekvngY7TqgBKYqgVfgC6fOg_vPdK8VYDA?e=xYrA0e' target='_blank'>DatFile]    |  
| Seismic GeoBody Identification  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/EvwMkQfKqJtOk6TP8U484yABSeCxjIL5gojWwqWSnMDeVg?e=NhbRWP' target='_blank'>DatFile]    |  
| Inversion (Reflectivity Estimation)  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/En2b7nDlY6BEn5tKXdcbi8oBTtO8CDRcir1IgGsnCYUeYw?e=dyTfnh' target='_blank'>DatFile]    |  
| Signal Processing (Denoise)   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/EnUGPcGo-hFFhrr2T4-wvSIB4KCQQJphdONXvaO1FOr_WA?e=rP057b' target='_blank'>DatFile]    |  
| Interpolation                 |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EWyYd0lXhfxOgffJIz5ICEUBRB_IqkbPoF1PQttUAfDLaQ?e=lR9qre' target='_blank'>DatFile]    |

# :neckbeard: Quick Guide

## Installation

To prepare the environment, please follow the following instructions.
```shell
# create virtual environment
conda create -n SFM python=3.9.12
conda activate SFM

# install pytorch
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# install other requirements
pip install -r requirements.txt

# if you want to visualize the results as shown in SFM-Finetune/Application/visualization.ipynb
pip install jupyter notebook
python -m ipykernel install --user --name=SFM --display-name="Python (SFM)"
```
## Download Dataset & Model

Place the downloaded dataset and model in the corresponding folder.
```shell
cd SFM-Pretrain
mkdir output_dir
# Download model and place it in folder SFM-Pretrain/output_dir
cd output_dir 
```

Download the Pretrain data zip file in ```Data``` folder.
```shell
# First execute merge
zip -s 0 mae_data_more.zip --out pretrain.zip
# Unzip the merged compressed file
unzip pretrain.zip
```

Download the DownStream Task data zip file in ```Data``` folder.
```shell
cd Data
# Download the DownStream Task data zip file in Data file
unzip *.zip
```
## Facies Example

### Download the DownStream Facies Task model [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EcK3TARvKDdCmvIT1lztxtEBSJqhMZmYuT7XWIG1nnT9jg?e=ly9rJh' target='_blank'>ckpt] and place it in folder SFM-Finetune/Application/Facies/SFM-Finetune/
 
#### Download the DownStream Facies Data [<a href='https://mailustceducn-my.sharepoint.com/:f:/g/personal/hanlins_mail_ustc_edu_cn/ElUKdIW6VhZOrekvngY7TqgBKYqgVfgC6fOg_vPdK8VYDA?e=xYrA0e' target='_blank'>DatFile] and place it in folder Data/ then ```unzip *.zip```

```shell

cd SFM-Finetune/Application
#Use jupyter notebbok to open visualization.ipynb
jupyter notebook
#open the folder Application and then open the file visualization.ipynb

```

<br>
<div>
# License
This project is released under the [MIT license](LICENSE).

