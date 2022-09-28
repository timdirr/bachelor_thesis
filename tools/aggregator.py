import os
import numpy as np
import pandas as pd
import json

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags
    out = defaultdict(list)
    steps = []


    for tag in tags:
        if tag.startswith("val/mIoU"):
            steps = [e.step for e in summary_iterators[0].Scalars(tag)]
            # for e in summary_iterators[0].Scalars(tag):
            #     print(e)
            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                assert len(set(e.step for e in events)) == 1
                out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    # for v in values:
    #     print(len(v))
    np_values = np.array(values)
    np_values = np.swapaxes(np_values, 1, 2)
    new_values = np.zeros((1, np_values.shape[1], 2))
    for idx in range(0, np_values.shape[1]):
        idx_1 = np.argmax(np_values[0][idx])
        np_values[0][idx][idx_1] = 0
        idx_2 = np.argmax(np_values[0][idx])

        new_values[0][idx][0] = steps[idx_1]
        if steps[idx_1] != 300000.:
            new_values[0][idx][1] = 300000.
        else:
            new_values[0][idx][1] = steps[idx_2]


    new_values = new_values.astype(int)
    print(new_values)
    out = {}
    for idx in range(0, len(dirs)):
        key = dirs[idx].replace("events.out.tfevents.", "")
        key = key.replace(".0", "")
        out[key] = [int(new_values[0][idx][0]),int(new_values[0][idx][1])]


    print(out)

    return out
    # for index, tag in enumerate(tags):
    #     df = pd.DataFrame(np_values[index], index=["best", "second"], columns=dirs)
    #     df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path1 = "/data/Code/timformer/tf_sum/10ksteps"
    o1 = to_csv(path1)
    path2 = "/data/Code/timformer/tf_sum/16ksteps"
    o2 = to_csv(path2)
    combined = {}
    for key, value in o1.items():
        combined[key] = value
    for key, value in o2.items():
        combined[key] = value
    print(combined)
    with open("/data/Code/timformer/tf_sum/results.json", "w") as write_file:
        json.dump(combined, write_file, indent=4)