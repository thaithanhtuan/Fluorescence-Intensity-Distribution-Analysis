# Fluorescence-Intensity-Distribution-Analysis
This github is generate for the paper "Fluorescence Intensity Distribution Analysis Using Instance Segmentation for Enhanced Cell Phenotyping in Microscopy Images"

<div align="center">
  
  <img src="image/Flow diagram New.drawio.png" width="100%" />
	<p>Pipeline for cell phenotyping using fluorescence microscopy images.</p>
</div>


# Dependency
Package            Version
------------------ --------------------
certifi            2023.5.7
charset-normalizer 3.1.0
colorama           0.4.6
contourpy          1.0.7
cycler             0.11.0
filelock           3.12.0
fonttools          4.39.4
idna               3.4
imgviz             1.7.3
Jinja2             3.1.2
joblib             1.3.1
kiwisolver         1.4.4
labelme            5.2.1
labelme2yolo       0.1.2
MarkupSafe         2.1.2
matplotlib         3.7.1
mpmath             1.3.0
natsort            8.4.0
networkx           3.1
numpy              1.24.3
opencv-python      4.7.0.72
packaging          23.1
pandas             2.0.2
Pillow             9.5.0
pip                23.1.2
psutil             5.9.5
py-cpuinfo         9.0.0
pyparsing          3.0.9
PyQt5              5.15.9
PyQt5-Qt5          5.15.2
PyQt5-sip          12.12.2
python-dateutil    2.8.2
pytz               2023.3
PyYAML             6.0
QtPy               2.3.1
requests           2.31.0
scikit-learn       1.2.2
scipy              1.10.1
seaborn            0.12.2
setuptools         67.7.2
six                1.16.0
sympy              1.12
termcolor          2.3.0
thop               0.1.1.post2209072238
threadpoolctl      3.2.0
torch              2.0.1
torchaudio         2.0.2+cu117
torchvision        0.15.2
tqdm               4.65.0
typing_extensions  4.6.2
tzdata             2023.3
ultralytics        8.1.14
urllib3            2.0.2
wheel              0.40.0
yolo2labelme       0.0.4


# Pretrained model of KG_Instance_Segmentation
You can download pretrained model from [here](https://drive.google.com/file/d/1jXbG1Rg0hxgYXDVMxa0vsWPSAK6He8vd/view?usp=sharing)): 

# How to run the code


Download the project KG_Instance_Segmentation: 

Change the test.py from this github. This test.py already modified to output the phenotyping features.


# Calculate Around The Cell Center

<div align="center">
  	<img src="image/Calculate Around The Cell Center.drawio.png" width="100%" />
	<p>Training log of YOLO.</p>
</div>

# Calculate Calculate Away From The Cell Center

<div align="center">
  	<img src="image/Calculate Away From The Cell Center.drawio.png" width="100%" />
	<p>Predict result on val set.</p>
</div>

# Calculate Cell Phenotyping

<div align="center">
  	<img src="image/Cell Phenotyping.drawio.png" width="100%" />
	<p>Predict result on val set.</p>
</div>


# Acknowledgement
This work is largely based on the PyTorch implementation of KG_Instance_Segmentation. We are extremely grateful for their public implementation.
[https://github.com/yijingru/KG_Instance_Segmentation](https://github.com/yijingru/KG_Instance_Segmentation)

# Citation
If this code helps your research, please cite our paper:

	@inproceedings{thaiplantphenomics,
		title={Fluorescence Intensity Distribution Analysis Using Instance Segmentation for Enhanced Cell Phenotyping in Microscopy Images},
		author={Thanh Tuan Thai, Giang Van Vu, Deokyeol Jeong, Soo Rin Kim, Jisoo Kim, and Jinhyun Ahn},
		booktitle={},
		pages={},
		year={}
	}
