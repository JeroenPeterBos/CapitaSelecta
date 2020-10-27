# CapitaSelecta
Repository for research into DynamicMultiView models for my Capita Selecta assignment at the University of Twente.

Run `dmv.main.py`
```
usage: parser.py [-h] [--data DATA] [--logs LOGS] [--category CATEGORY]
                 [--img-size IMG_SIZE IMG_SIZE IMG_SIZE]
                 [--batch-size BATCH_SIZE BATCH_SIZE]
                 [--cache-imgs CACHE_IMGS] [--max-imgs MAX_IMGS]
                 [--max-epochs MAX_EPOCHS] [--replication REPLICATION]
                 [--shuffle-size SHUFFLE_SIZE]
                 model run_id

Run a dynamic multi view research experiment.

positional arguments:
  model                 The python directory path to the model to run the
                        experiments on.
  run_id                The unique describing id for the type of run.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to the MURA dataset.
  --logs LOGS           The base path to the logs.
  --category CATEGORY   The category to use for training, excluding all the
                        others.
  --img-size IMG_SIZE IMG_SIZE IMG_SIZE
                        The image resolution to train the network on.
  --batch-size BATCH_SIZE BATCH_SIZE
                        The single and multi view batch_size.
  --cache-imgs CACHE_IMGS
                        Whether to cache the images just before augmentation.
  --max-imgs MAX_IMGS   The maximum number of images a study is allowed to
                        contain.
  --max-epochs MAX_EPOCHS
                        The maximum number of epochs to train the model for if
                        early stopping doesn't kick in.
  --replication REPLICATION
                        The number of times to replicate the experiment.
  --shuffle-size SHUFFLE_SIZE
                        Shuffle size in samples. (-1 for full shuffle)
```