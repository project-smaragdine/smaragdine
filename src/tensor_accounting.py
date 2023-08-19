import json
import os
import sys
import zipfile

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from tqdm import tqdm

# code to create minimal event traces (just enough to align) from the timeline
def get_inputs(e):
    inputs = []
    for key in e['args']:
        if 'input' in key:
            inputs.append(e['args'][key])
    if inputs:
        return '@' + ';'.join(inputs)
    else:
        return ''

def create_event(ts, dur, device, pid, op):
    return {
        'ts': ts,
        'dur': dur,
        'device': device,
        'pid': pid,
        'op': op
    }

def create_trace(events):
    return list(map(lambda e: create_event(**e), events))

def get_traces(timeline):
    metadata = [e for e in timeline['traceEvents'] if e['ph'] == 'M']
    devices = {}
    # TODO: find a way to map this more gracefully
    for device in metadata:
        device_name = device['args']['name']
        if ':GPU:' in device_name:
            devices[device['pid']] = 'GPU:' + device_name.split(':GPU:')[1].split('/')[0].split(' ')[0]
        elif ':CPU:' in device_name:
            devices[device['pid']] = 'CPU:' + device_name.split(':CPU:')[1].split('/')[0].split(' ')[0]

    # pull out all eXecution events and turn them into events
    events = create_trace({
        'ts': e['ts'],
        'dur': e['dur'],
        'device': devices[e['pid']],
        'pid': e['pid'],
        'op': e['args']['name'] + get_inputs(e)
    } for e in timeline['traceEvents'] if e['ph'] == 'X')
    return events

# code to segment (mark) traces
def create_segment():
    return {'entry': set(), 'exit': set()}

EXCLUDED_OPS = ['_SOURCE', 'cuStreamSynchronize']

# TODO: there's a way to parallelize the footprint generation if we can shape it
# as {'start': ts, 'end': ts, 'ops': set} here
def trace_to_segments(trace):
    segments = {}
    # for each event, add the event op to the entry of its start timestamp and
    # the exit of its end timestamp
    for event in trace:
        if event['op'] in EXCLUDED_OPS:
            continue
        if event['device'] not in segments:
            segments[event['device']] = {}

        trace = segments[event['device']]
        start = event['ts']
        end = event['ts'] + event['dur']
        if start not in trace:
            trace[start] = create_segment()
        if end not in trace:
            trace[end] = create_segment()
        trace[start]['entry'].add(event['op'])
        trace[end]['exit'].add(event['op'])
    return segments

# code to generate footprints
def generate_footprint(trace, power):
    energy = {}
    runtime = {}

    # TODO: sigh. one day i'll figure out how to write this optimally
    trace_ts = list(trace)
    trace_ts.sort()
    t_i = 0

    power_ts = list(power)
    power_ts.sort()
    p_i = 0

    ts = list(set(power_ts + trace_ts))
    ts.sort()

    start, end = min(trace_ts), max(trace_ts)
    ops = set()
    i = ts.index(start)

    while start < end:
        current = ts[i + 1]
        while p_i != len(power_ts) - 1 and power_ts[p_i + 1] < current:
            p_i += 1
        while t_i != len(trace_ts) - 1 and trace_ts[t_i + 1] < current:
            t_i += 1

        events = trace[trace_ts[t_i]]
        ops |= events['entry']
        if len(ops) != 0:
            # units from the trace are in micros but power is in watts
            r = (current - start) / 1000000
            p = power[power_ts[p_i]]
            for op in ops:
                if op not in energy:
                    energy[op] = 0
                    runtime[op] = 0
                runtime[op] += r
                energy[op] += p * r / len(ops)
            ops -= events['exit']

        i += 1
        start = current
    return {'energy': energy, 'runtime': runtime}

def parse_args():
    """ Parses accounting arguments. """
    parser = ArgumentParser()
    parser.add_argument(
        dest='files',
        nargs='*',
        default=None,
        help='sessions to align',
    )
    return parser.parse_args()

def main():
    args = parse_args()

    for data_path in args.files:
        print(f'accounting {data_path}')

        print('loading energy data')
        f = zipfile.ZipFile(os.path.join(data_path, 'eflect-data.zip'))
        rapl = pd.read_csv(f.open('rapl.csv')).dropna()
        nvml = pd.read_csv(f.open('nvml.csv')).dropna()

        # TODO: this is an oversight by me. these are power state, not
        # energy. it will be energy in the footprint
        power = pd.concat([
            rapl.assign(device='CPU:' + rapl.socket.astype(str)).groupby(['timestamp', 'device']).energy.sum(),
            nvml.assign(device='GPU:0').groupby(['timestamp', 'device']).energy.sum(),
            # nvml.assign(device='GPU:' + nvml.device_id).groupby(['timestamp', 'device']).energy.sum(),
        ]).reset_index()
        power['power'] = power.energy
        power['ts'] = pd.to_datetime(power.timestamp).astype(np.int64) // 1000
        # some ugly magic to create a dict<device, dict<timestamp, power>>
        power = power.groupby('device')[['ts', 'power']].apply(
            lambda s: s.set_index('ts').power.to_dict()).to_dict()

        print('loading traces')
        traces = np.sort(list(map(
            lambda s: int(s.split('-')[1].split(r'.')[0]),
            filter(
                lambda s: 'timeline' in s and 'json' in s,
                os.listdir(data_path)
            )
        )))
        traces = {
            epoch: get_traces(json.load(open(os.path.join(
                data_path,
                'timeline-{}.json'.format(epoch)
            )))) for epoch in tqdm(traces)
        }
        segments = {epoch: trace_to_segments(traces[epoch]) for epoch in traces}

        def get_runtime(device_traces):
            ts = [ts for trace in device_traces.values() for ts in trace]
            return (max(ts) - min(ts)) / 1000000
        runtime = pd.Series({
            epoch: get_runtime(segments[epoch]) for epoch in segments
        })

        devices = power.keys()
        footprint = {epoch: {
            device: generate_footprint(
                segments[epoch][device], power[device]) for device in devices
        } for epoch in tqdm(segments)}

        df = []
        for epoch in footprint:
            for device in footprint[epoch]:
                s = pd.DataFrame(footprint[epoch][device])
                s.index.name = 'operation'
                df.append(s.assign(epoch=epoch, device=device))
        footprint = pd.concat(df)
        footprint = footprint.reset_index().set_index(['device', 'epoch', 'operation'])

        print(runtime.agg(('mean', 'std')))
        print(footprint.groupby(['device', 'operation']).sum().sort_values(by='energy', ascending=False).head(20))

        runtime.to_csv(os.path.join(data_path, 'runtime.csv'))
        footprint.to_csv(os.path.join(data_path, 'footprint.csv'))

if __name__ == '__main__':
    main()
