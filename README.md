# flower_classifier
Developing an AI application  that will Train to classify Flower types with up to 102 classes of species. Using pretrained Model VGG  and extend it with your own classifier. user will have possibility to choose different vgg model in application

# Dataset
The model can be trained on any dataset user want but need to be relevant for Flower classification otherwise results wont be as desired. Please use recommeneded dataset provided in following link: https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz OR
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

# Instruction
Project has 2 files predict.py and train.py, as name suggestion one is for training model and other one is to predict as inference on trained model. Following are instruction how to use command line argument for each

## Training
One must execute training before trying to predict something as model does not exist any more
usage: Train on Images [-h] [--arch ARCH] [--learning_rate LEARNING_RATE]
                       [--hidden_units [HIDDEN_UNITS ...]] [--epochs EPOCHS]
                       [--gpu]
                       directory

Train on Images to classify Flower types with unfreezing last 8 layersof the
base model

positional arguments:
  directory             Path to the folder of flower images

options:
  -h, --help            show this help message and exit
  --arch ARCH           CNN Model architecture to useNote: only allowed vgg13,
                        vgg16, vgg19
  --learning_rate LEARNING_RATE
                        Learning rate
  --hidden_units [HIDDEN_UNITS ...]
                        Hidden units separated by space e.g. 1000 512 default
                        [4096,1000]
  --epochs EPOCHS       Number of epochs default 5
  --gpu                 Use GPU for training

## Predict
following argument description to predict for given input image. If user wants to see Bar graph of requested top probabilities then path to the cat_to_name.json is necessary otherwise tool just shows logs with 5 highest probability

usage: Predict Image [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                     [--gpu]
                     image_path checkpoint

Predict image to classify Flower types

positional arguments:
  image_path            Path to image to predict must be label.jpg e.g.
                        mexican aster.jpg
  checkpoint            Path to model checkpoint file .pthNote: only supports
                        vgg13, vgg16, vgg19

options:
  -h, --help            show this help message and exit
  --top_k TOP_K         Number of top classes to show default 5
  --category_names CATEGORY_NAMES
                        JSON file for category names
  --gpu                 Use GPU for inference
