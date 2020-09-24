import time
import pandas as pd
import tensorflow as tf


def _benchmark(dataset, num_epochs=3, sleep=0.0):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            if sleep > 0:
                time.sleep(sleep)
    tf.print("Execution time:", (time.perf_counter() - start_time) / num_epochs)


def mura_meta(file):
    df = pd.read_csv(file, header=None, names=['full_path'])

    components = df['full_path'].str.split('/')
    df['folder'] = components.str[:-1].str.join('/')
    df['file'] = components.str[-1]

    df = df.groupby('folder').agg({'file': lambda x: list(x)}).reset_index()

    components = df['folder'].str.split('/')
    df['location'] = components.str[2].str.split('XR_').str[1].str.title()
    df['patient'] = components.str[3].str.replace('patient', '')
    df['session'] = components.str[4].str.split('_').str[0].str.replace('study', '')
    df['study'] = df['patient'] + '-' + df['session']

    df['label'] = ((components.str[4]).str.contains('positive')).astype(int)
    df['index'] = df.index
    return df