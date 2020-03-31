# Proportional Ink

This project aims to identify barcharts that violate principle of [proportional ink](https://callingbullshit.org/tools/tools_proportional_ink.html). As part of this project, a framework is developed to automatically extract numeric axes labels of barcharts. 

Given a path to a folder of images (image_dir), this framework extracts individual graphs from compound figures (Figure Separator), identifies bar charts among them (Bar Classifier) and extracts labels for barcharts (Label Extractor). The outputs are annotated images in a specified folder(image_output_dir) and labels as csv file in a specified folder(csv_output).

For separating figures, we used open source application [Compound Figure Separator](https://github.com/apple2373/figure-separator) which is an implementation using a covolutional neural network (CNN). For classifying bar charts,we applied transfer learning and used RESNET-50 pre-trained model as weight intialization scheme. This model is integrated with a 3 layer neural network built using Keras. For extracting labels, we used open source repository [Scatteract](https://github.com/bloomberg/scatteract).Scatteract is a framework to automatically extract data from the image of scatter plots. We modified the code to extract labels from bar charts.


### Requirements:
tensorflow (tested on 0.14.0) <br />
scipy (tested on 0.17.1) <br />
scikit-learn (tested on 0.17.1) <br />
pandas (tested on 0.18.1) <br />
Pillow (PIL) (tested on 3.2.0) <br />
numpy (tested on 1.10.4) <br />
opencv-python (cv2)  (tested on 2.4.10) <br />
matplotlib  (tested on 1.5.1) <br />
runcython (tested on 0.25) <br />
pyocr  (tested on 0.6) <br />
tesseract-ocr (tested on 4.0) <br/>
keras (tested on 2.2.4) <br/>
pytesseract (tested on 0.3.2)

On Python 2.7+, the following compatibility module is also required:

backports.functools_lru_cache

### Other Requirement: Pre-trained models
Download pre-trained models directory: [pre-trained models](https://drive.google.com/drive/folders/1O0ad9ZTW8Q67_6GI7kgn1RSBS27m3pir?usp=sharing).

Copy labels_v1 and ticks_v1 to ouput folder <br />
Copy classifybars to bc_models folder <br />
Copy figure-separation-model to figure-separator/data folder

## How to use:

### Extract labels 

    $ python label_extractor.py --image_dir data/plot_rt/images --image_output_dir image_output --csv_output_dir csv_output

### Test pipeline

    $ python label_extractor.py --image_dir data/plot_rt/images --image_output_dir image_output --csv_output_dir csv_output --true_labels data/plot_rt/true_labels.csv
    

### Acknowledgements

[Compound Figure Separator](https://github.com/apple2373/figure-separator) and [Scatteract](https://github.com/bloomberg/scatteract) are used in this pipeline. Thank you very much for the authors of these repositories.
    





