""" processing and virtualization code for a DataSet using pandas """
import os

from argparse import ArgumentParser
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pandas import to_datetime

from protos.sample_pb2 import DataSet

# processing helpers
INTERVAL = '4ms'
WINDOW_SIZE = '101ms'


def bucket_timestamps(timestamps, interval=INTERVAL):
    """ Floors a series of timestamps to some interval for easy aggregates. """
    return to_datetime(timestamps).dt.floor(interval)


def max_rolling_difference(df, window_size=WINDOW_SIZE):
    """ Computes a rolling difference of points up to the window size. """
    values = df - df.rolling(window_size).min()

    timestamps = df.reset_index().timestamp.astype(int) / 10**9
    timestamps.index = df.index
    timestamps = timestamps - timestamps.rolling(window_size).min()

    return values, timestamps


# cpu jiffies processing
def parse_cpu_samples(samples):
    """ Converts a collection of CpuSample to a DataFrame. """
    records = []
    for sample in samples:
        for stat in sample.reading:
            records.append([
                sample.timestamp,
                stat.cpu,
                stat.socket,
                stat.user,
                stat.nice,
                stat.system,
                stat.idle,
                stat.iowait,
                stat.irq,
                stat.softirq,
                stat.steal,
                stat.guest,
                stat.guest_nice
            ])
    df = pd.DataFrame(records)
    df.columns = [
        'timestamp',
        'cpu',
        'socket',
        'user',
        'nice',
        'system',
        'idle',
        'iowait',
        'irq',
        'softirq',
        'steal',
        'guest',
        'guest_nice'
    ]
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    return df


ACTIVE_JIFFIES = [
  'cpu',
  'user',
  'nice',
  'system',
  'irq',
  'softirq',
  'steal',
  'guest',
  'guest_nice',
]


def process_cpu_data(df):
    """ Computes the cpu jiffy rate of each bucket """
    df['jiffies'] = df[ACTIVE_JIFFIES].sum(axis=1)
    df.timestamp = bucket_timestamps(df.timestamp)

    jiffies = df.groupby(['timestamp', 'socket', 'cpu']).jiffies.min().unstack().unstack()
    jiffies, ts = max_rolling_difference(jiffies)
    jiffies = jiffies.stack().stack().reset_index()
    jiffies = jiffies.groupby(['timestamp', 'socket', 'cpu']).sum().unstack()
    jiffies = jiffies.div(ts, axis=0).stack()[0]
    jiffies.name = 'jiffies'

    return jiffies


def cpu_samples_to_df(samples):
    """ Converts a collection of CpuSamples to a processed DataFrame. """
    return process_cpu_data(parse_cpu_samples(samples))


# task jiffies processing
def parse_task_samples(samples):
    """ Converts a collection of ProcessSamples to a DataFrame. """
    records = []
    for sample in samples:
        for stat in sample.reading:
            records.append([
                sample.timestamp,
                stat.task_id,
                stat.name if stat.HasField('name') else '',
                stat.cpu,
                stat.user,
                stat.system
            ])
    df = pd.DataFrame(records)
    df.columns = [
        'timestamp',
        'id',
        'thread_name',
        'cpu',
        'user',
        'system',
    ]
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    return df


def process_task_data(df):
    """ Computes the app jiffy rate of each bucket """
    df['jiffies'] = df.user + df.system
    df.timestamp = bucket_timestamps(df.timestamp)

    cpus = df.groupby(['timestamp', 'id']).cpu.max()
    jiffies, ts = max_rolling_difference(df.groupby([
        'timestamp',
        'id'
    ]).jiffies.min().unstack())
    jiffies = jiffies.stack().to_frame()
    jiffies = jiffies.groupby([
        'timestamp',
        'id',
    ])[0].sum().unstack().div(ts, axis=0).stack().to_frame()
    jiffies['cpu'] = cpus
    jiffies = jiffies.reset_index().set_index(['timestamp', 'id', 'cpu'])[0]
    jiffies.name = 'jiffies'

    return jiffies


def task_samples_to_df(samples):
    """ Converts a collection of ProcessSamples to a processed DataFrame. """
    return process_task_data(parse_task_samples(samples))


# nvml processing
def parse_nvml_samples(samples):
    """ Converts a collection of RaplSamples to a DataFrame. """
    records = []
    for sample in samples:
        for reading in sample.reading:
            records.append([
                sample.timestamp,
                reading.index,
                reading.bus_id,
                reading.power_usage,
            ])
    df = pd.DataFrame(records)
    df.columns = [
        'timestamp',
        'device_index',
        'bus_id',
        'power_usage',
    ]
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    return df


def process_nvml_data(df):
    """ Computes the power of each 50ms bucket """
    df.timestamp = bucket_timestamps(df.timestamp)
    df = df.groupby(['timestamp', 'device_index', 'bus_id']).power_usage.min()
    df.name = 'power'
    return df


def nvml_samples_to_df(samples):
    """ Converts a collection of NvmlSamples to a processed DataFrame. """
    return process_nvml_data(parse_nvml_samples(samples))


# rapl processing
WRAP_AROUND_VALUE = 16384


def parse_rapl_samples(samples):
    """ Converts a collection of RaplSamples to a DataFrame. """
    records = []
    for sample in samples:
        for reading in sample.reading:
            records.append([
                sample.timestamp,
                reading.socket,
                reading.cpu,
                reading.package,
                reading.dram,
                reading.gpu
            ])
    df = pd.DataFrame(records)
    df.columns = [
        'timestamp',
        'socket',
        'cpu',
        'package',
        'dram',
        'gpu'
    ]
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    return df


def maybe_apply_wrap_around(value):
    """ Checks if the value needs to be adjusted by the wrap around. """
    if value < 0:
        return value + WRAP_AROUND_VALUE
    else:
        return value


def process_rapl_data(df):
    """ Computes the power of each bucket """
    df.timestamp = bucket_timestamps(df.timestamp)
    df = df.groupby(['timestamp', 'socket']).min()
    df.columns.name = 'component'

    energy, ts = max_rolling_difference(df.unstack())
    energy = energy.stack().stack().apply(maybe_apply_wrap_around)
    energy = energy.groupby([
        'timestamp',
        'socket',
        'component'
    ]).sum()
    power = energy.div(ts, axis=0).dropna()
    power.name = 'power'

    return power


def rapl_samples_to_df(samples):
    """ Converts a collection of RaplSamples to a processed DataFrame. """
    return process_rapl_data(parse_rapl_samples(samples))


# virtualization
def virtualize_jiffies(tasks, cpu):
    """ Returns the ratio of the jiffies with attribution corrections. """
    activity = (tasks / cpu).dropna().replace(np.inf, 1).clip(0, 1)
    activity = activity[activity > 0]
    activity.name = 'activity'
    return activity


def virtualize_with_activity(activity, power):
    """ Computes the product of the data across shared indices. """
    try:
        df = activity * power
        df.name = 'power'
        return df
    except Exception as e:
        # TODO(timur): sometimes the data can't be aligned and i don't know why
        idx = list(set(activity.index.names) & set(power.index.names))
        print('data could not be directly aligned: {}'.format(e))
        print('forcing merge on {} instead'.format(idx))
        power = pd.merge(
            activity.reset_index(),
            power.reset_index(),
            on=['timestamp']
        ).set_index(idx)
        power = power.activity * power.power
        power.name = 'power'
        return power


def virtualize_nvml_energy(nvml):
    """ Returns the product of energy and activity. """
    df = nvml_samples_to_df(nvml)
    df = df[df > 0].dropna() / 1000
    return df


def virtualize_rapl_energy(tasks, cpu, rapl):
    """ Returns the product of power and activity by socket. """
    activity = virtualize_jiffies(tasks, cpu)
    rapl = rapl_samples_to_df(rapl)

    power = virtualize_with_activity(activity, rapl)
    power = power[power > 0].dropna() / 1000000

    return energy


def virtualize_data(data):
    """ Produces energy virtualizations from a data set. """
    print('virtualizing application activity...')
    tasks = task_samples_to_df(data.process)
    cpu = cpu_samples_to_df(data.cpu)

    virtualization = {}

    if len(data.nvml) > 0:
        print('accounting nvml...')
        virtualization['nvml'] = virtualize_nvml_energy(data.nvml)
    if len(data.rapl) > 0:
        print('accounting rapl...')
        virtualization['rapl'] = virtualize_rapl_energy(tasks, cpu, data.rapl)

    return virtualization


# cli to process globs of files
def parse_args():
    """ Parses virtualization arguments. """
    parser = ArgumentParser()
    parser.add_argument(
        dest='files',
        nargs='*',
        default=None,
        help='files to process',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        dest='output',
        default=None,
        help='directory to write the processed data to',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for file in args.files:
        with open(file, 'rb') as f:
            data = DataSet()
            data.ParseFromString(f.read())

        if args.output:
            if os.path.exists(args.output) and not os.path.isdir(args.output):
                raise RuntimeError(
                    'output target {} already exists and is not a directory; aborting'.format(args.output))
            elif not os.path.exists(args.output):
                os.makedirs(args.output)

            path = os.path.join(args.output, os.path.splitext(
                os.path.basename(file))[0]) + '.zip'
        else:
            path = os.path.splitext(file)[0] + '.zip'
        print('virtualizing data from {}'.format(file))
        footprints = virtualize_data(data)

        # TODO: this only spits out a single file. we should be able to write
        #   multiple files to the archive, but maybe not with pandas
        with ZipFile(path, 'w') as archive:
            for key in footprints:
                archive.writestr('{}.csv'.format(key), footprints[key].to_csv())


if __name__ == '__main__':
    main()
