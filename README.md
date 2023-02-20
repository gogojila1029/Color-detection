# Color-detection
<img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=Keras&logoColor=white"/></a>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/></a>
## Introduction

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This is an AI project to detect object's colors, which is captured by an object detection algorithm called Yolov3.<br>  
We used KNN(K-Nearest Neighbor) algorithm to extract the most frequent pixels that represent a color and record them on a notepad(training3.txt).
These pixels on the notepad will be used to train knn algorithm, and this algorithm will predict a specific color of a frame that is captured from a webcam using OpenCV library.

## Demo
![result1](https://user-images.githubusercontent.com/99879082/212631803-50a4a45e-44d4-45ba-9a7c-ccc0fe194226.PNG)

![result2](https://user-images.githubusercontent.com/99879082/212631812-c34e9821-a9d9-4868-b151-fd02971c9a81.PNG)

## Theory

## Requirements

This project requires several libraries:

- [Numpy-1.19.5](https://numpy.org/) 
- [h5py-2.10.0](https://www.h5py.org/)
- [pandas-1.2.3](https://pandas.pydata.org/) 
- [torch-1.9.0+cu111](https://pytorch.org/get-started/previous-versions/)
- [opencv-python-4.5.1.48](https://opencv.org/)
- [pillow-9.0.1](https://pypi.org/project/Pillow/9.0.1/)
- [Python-3.8.0](https://www.python.org/downloads/release/python-380/)

>Please install all the libralies using requirements.txt


## Installation

1. First, clone the repository

2. From [data](https://drive.google.com/drive/folders/1_GnazrVC9MHOFEcqacNN1MxGSvIaqhFM?usp=sharing) link, download two folders

   ```
   data
   cfg
   ```
3. Put those two files into the repository

## Run
```
1.For training:
   - Run "color_histogram_feature_extraction.py"\
   
2.For testing:
   - Run "cam.py"
```

## About Us
![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=gogojila1029&show_icons=true&theme=radical)
[//]![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=Aiuces&show_icons=true&theme=radical)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><br>Copyright 2022. Suhwan Lim. all rights reserved.</br>
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
