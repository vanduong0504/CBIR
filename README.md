<h1 align="center"> GENERATIVE ADVERSARIAL NETWORKS </h1>

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<h2> :pencil: About the project </h2>

This project aims to create an sample image retrieval system. Using **VGG16** pretrained model and small Felidae dataset in [kaggle](https://www.kaggle.com/vishweshsalodkar/wild-animals) for demonstrating.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :floppy_disk: Dataset </h2>

For custom dataset, follow this structure:

    📂 your dataset
    |─── class-1
    |      +-- *.jpg
    |─── class-2
    |      +-- *.jpg
    
    # For example
    📂 Dog-and-Cat
    |─── class-1
    |      +-- 001.jpg
    |      +-- 002.jpg
    |─── class-2
    |      +-- 001.jpg
    |      +-- 002.jpg        

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :scroll: Usage </h2>
<ul>
<li> First you need to install some helpful packages:</li>
    
`pip install requirements.txt`

<li> Using extract.py to extract features from your dataset and store in .pkl file: </li>

`python extract.py --dataroot "path/to/dataroot"`

<li> Using main.py to calculate the similarity between querry image and dataroot and select top K : </li>

`python main.py --pkl_dir "path/to/pkl_dir" --querry "path/to/querry image" --k 10`

<h2> :pushpin: Note </h2>

`main.py` will also save an image which ploted by `matplotlib`. You can see **imgs/sample.jpg**
