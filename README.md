<p align="center" width="100%">
<img src="SeismicPretrainedModel.png"  width="80%" height="80%">
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

### 🌟 Seismic Foundation Model (SFM)


 As shown in this workflow figure, SFM can provide services for downstream tasks including seismic image classification and geobody identification. Additionally, we test the model's performance in regression tasks, specifically in signal processing (i.e. denoising), inversion (i.e. reflectivity estimation), and interpolation.


# 🌟 News
* **2023.8.7:** Github Repository Initialization (copy from Meta-Transformer). The paper and model will be release soon. ⌛⌛⌛


# :rocket: Model Zoo & Data Release

<!-- <details> -->
<summary> Open-source Pretrained Models </summary>
<br>
<div>

|    Model   |      Pretraining Size      |  Download |
|------------|:--------------------------:|:----------:|
| SFM-Base   |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ef1xhsxytZRNjYiXJJGQAJEB2S9Lj76yTQOKF0EYUeCUHg' target='_blank'>ckpt]    |  
| SFM-Base-512   |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/ES-iLYELZq1IiOor3LgDGPsBIYXGs98BeBpquW4srfJz_g' target='_blank'>ckpt]    |  
| SFM-Large  |         224 × 224          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/Ec7ZqiAwdvtGpN5vasPnalwBZGl6z0fkaS9hFTWQ6ZVjMQ' target='_blank'>ckpt]    |
| SFM-Large-512  |         512 × 512          | [<a href='https://mailustceducn-my.sharepoint.com/:u:/g/personal/hanlins_mail_ustc_edu_cn/EXwVJmoAmRJFoenCvsIsolQBhcmlL3L_tzV5Ubm5nXsNEw' target='_blank'>ckpt]    |    

<summary> Open-source Training & DownStream Fine-tune Task Data</summary>
<br>
<div>
|    Task   |      Size      |  Download |
|:------------------:|:--------------------------:|:----------:|
| PreTrain   |         224 × 224          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |  
| Seismic Facies Classification   |         768 × 768          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |  
| Seismic GeoBody Identification  |         224 × 224          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |  
| Inversion (Reflectivity Estimation)  |         224 × 224          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |  
| Signal Processing (Denoise)   |         224 × 224          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |  
| Interpolation                 |         224 × 224          | [<a href='https://github.com/shenghanlin/' target='_blank'>DatFile]    |

  

<br>
<div>
# License
This project is released under the [MIT license](LICENSE).

