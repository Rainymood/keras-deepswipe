﻿# Keras DeepSwipe

End-to-end deep learning model for natural gesture recognition powered by Keras.

# TODO

- [x] Copy preprocessed data to `/data` (Wed Mar 27 14:24:26 CET 2019)
- [x] Move hardcoded input size to config.json `(NUM_FRAMES, HEIGHT, WIDTH, ...)` in `build_model()`  (Wed Mar 27 14:24:26 CET 2019)
- [x] Create `deepswipe_model.py` (Wed Mar 27 14:24:26 CET 2019)
- [x] Create `deepswipe_data_loader.py`
- [x] Create `deepswipe_trainer.py`
- [ ] Train model
- [ ] Create a data pipeline from raw data to processed data (get rid of files <10 frames, take last 10 frames)
- [ ] Refactor `config.trainer.crop : ['first','last','random']
- [ ] Gather new training data 

# Table of contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Running The Demo Project](#running-the-demo-project)
- [Comet.ml Integration](#cometml-integration)
- [Template Details](#template-details)
    - [Folder Structure](#folder-structure)
    - [Main Components](#main-components)
- [Acknowledgements](#acknowledgements)

# Installation

Clone the repository
```shell
git clone https://github.com/Rainymood/keras-deepswipe
cd keras-deepswipe
```
Set-up the virtual environment
```shell
virtualenv --system-site-packages -p python3 venv
```
Note that `/venv` is included in `.gitignore`. Now enter the virutal environment
```shell
. venv/bin/activate
```
Install dependencies
```shell
pip install -r requirements.txt
```

# Getting Started

In order to get started we have to:
1. Define a data loader class.
2. Define a model class that inherits from BaseModel.
3. Define a trainer class that inherits.
4. Define a configuration file with the parameters needed in an experiment.
5. Run the model using:
```shell
python main.py -c [path to configuration file]
```

# Running The Demo Project
A simple model for the mnist dataset is available to test.
To run the demo project:
1. Start the training using:
```shell
python main.py -c configs/simple_mnist_config.json
```
2. Start Tensorboard visualization using:
```shell
tensorboard --logdir=experiments/<YYYY-MM-DD>/simple_mnist/logs
```
*Note*: One has to fill in the appropriate date.

<div align="center">

<img align="center" width="600" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/Tensorboard_demo.PNG?raw=true">

</div>

# Running the main project

```shell
python main_deepswipe.py -c configs/deepswipe_config.json
```

And then for the visualisation you can run 

```shell
tensorboard --logdir=experiments/<YYYY-MM-DD>/deepswipe/logs
```

# Comet.ml Integration
Support for Comet.ml is integrated. This allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metrics.

Add your API key [in the configuration file](configs/simple_mnist_config.json#L15):

For example:  `"comet_api_key": "your key here"`

# Template Details

## Folder Structure

```
├── main.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── simple_mnist_model.py
│   └── i3d_inception.py
│   └── deepswipe_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── simple_mnist_trainer.py
│   └── deepswipe_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── simple_mnist_data_loader.py
│   └── deepswipe_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── simple_mnist_config.json
│   └── deepswipe_config.json
│
│
├── data                - this folder might contain the datasets of your project.
│
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```

## Main Components

### Models
You need to:
1. Create a model class that inherits from **BaseModel**.
2. Override the ***build_model*** function which defines your model.
3. Call ***build_model*** function from the constructor.

### Trainers
You need to:
1. Create a trainer class that inherits from **BaseTrainer**.
2. Override the ***train*** function which defines the training logic.

**Note:** To add functionalities after each training epoch such as saving checkpoints or logs for tensorboard using Keras callbacks:
1. Declare a callbacks array in your constructor.
2. Define an ***init_callbacks*** function to populate your callbacks array and call it in your constructor.
3. Pass the callbacks array to the ***fit*** function on the model object.

**Note:** You can use ***fit_generator*** instead of ***fit*** to support generating new batches of data instead of loading the whole dataset at one time.

### Data Loaders
You need to:
1. Create a data loader class that inherits from **BaseDataLoader**.
2. Override the ***get_train_data()*** and the ***get_test_data()*** functions to return your train and test dataset splits.

**Note:** You can also define a different logic where the data loader class has a function ***get_next_batch*** if you want the data reader to read batches from your dataset each time.

### Configs
You need to define a .json file that contains your experiment and model configurations such as the experiment name, the batch size, and the number of epochs.

### From Config
We can now load models without having to explicitly create an instance of each class. Look at:
1. from_config.py: this can load any config file set up to point to the right modules/classes to import
2. Look at configs/simple_mnist_from_config.json to get an idea of how this works from the config. Run it with:
```shell
python from_config.py -c configs/simple_mnist_from_config.json
```
3. See conv_mnist_from_config.json (and the additional data_loader/model) to see how easy it is to run a different experiment with just a different config file:
```shell
python from_config.py -c configs/conv_mnist_from_config.json
```

# Acknowledgements
This project builds off the [Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template#getting-started).
