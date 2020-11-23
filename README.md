# Welcome to Dynamic Dense
Repository for research into DynamicMultiView models for my Capita Selecta assignment at the University of Twente.

### Installation
Installing the code: `pip install --upgrade --force-reinstall git+https://github.com/JeroenPeterBos/CapitaSelecta.git`

### Replication
The folded datasets can be generated using the `GenerateFolds.ipynb` notebook. The results presented in my research can be replicated by running the following commands.

```
python -m dmv.final.main \
        --data <path to your folds> \
        --logs <path to your logs> \
        --shuffle-size 1000 \
        --img-size 128 128 3 \
        --max-imgs 12 \
        --learning-rate 0.00001 \
        --cache-data \
        --redirect-err \
        --tensorboard \
        --batch-size 16 16 \
        --folds 0 1 2 3 4 5 6 \
        single

python -m dmv.final.main \
        --data <path to your folds> \
        --logs <path to your logs> \
        --shuffle-size 1000 \
        --img-size 128 128 3 \
        --max-imgs 12 \
        --learning-rate 0.00001 \
        --cache-data \
        --redirect-err \
        --tensorboard \
        --batch-size 16 16 \
        --folds 0 1 2 3 4 5 6 \
        multi
```


### Options
The run options for `dmv.final.main` are:
```
usage: main.py [-h] [--data DATA] [--logs LOGS] [--folds FOLDS [FOLDS ...]]
               [--img-size IMG_SIZE IMG_SIZE IMG_SIZE]
               [--batch-size BATCH_SIZE BATCH_SIZE] [--cache-data]
               [--max-imgs MAX_IMGS] [--max-epochs MAX_EPOCHS]
               [--shuffle-size SHUFFLE_SIZE] [--redirect-err] [--tensorboard]
               [--learning-rate LEARNING_RATE]
               model_types

Run a dynamic multi view research experiment.

positional arguments:
  model_types           Which model types to run on, either 'multi' or
                        'single'.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to the MURA dataset.
  --logs LOGS           The base path to the logs.
  --folds FOLDS [FOLDS ...]
                        The folds to try to load.
  --img-size IMG_SIZE IMG_SIZE IMG_SIZE
                        The image resolution to train the network on.
  --batch-size BATCH_SIZE BATCH_SIZE
                        The single and multi view batch_size.
  --cache-data          Whether to cache the images just before augmentation.
  --max-imgs MAX_IMGS   The maximum number of images a study is allowed to
                        contain.
  --max-epochs MAX_EPOCHS
                        The maximum number of epochs to train the model for if
                        early stopping doesn't kick in.
  --shuffle-size SHUFFLE_SIZE
                        Shuffle size in samples. (-1 for full shuffle)
  --redirect-err        Redirect all the logs (also .cc/cpp) to a file, the
                        stderr will be hijacked so the stream will be gone.
  --tensorboard         Store results in tensorboard files
  --learning-rate LEARNING_RATE
                        The learning rate of the optimizer
```

### Exploration
The `dmv.main` module is more suitable for running the code in an exploratory fashion.