"""Positives known strictly before a sampled prediction time."""
import csv
import gzip
from pathlib import Path


def positive_events(paths, id_to_row, users):
    events = {user: {} for user in users}
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as source:
            for row in csv.DictReader(source):
                if row['user_id'] not in events:
                    continue
                if 'rating' in row and float(row['rating']) < 4:
                    raise ValueError(f'Nonpositive interaction in {path}')
                item, timestamp = id_to_row[row['parent_asin']], int(row['timestamp'])
                previous = events[row['user_id']].get(item)
                if previous is None or timestamp < previous:
                    events[row['user_id']][item] = timestamp
    return events


def known_before(events, user, timestamp):
    return {item for item, first in events.get(user, {}).items() if first < timestamp}


def prior_paths(data, split):
    data = Path(data)
    paths = [data / 'interactions_train.csv.gz']
    if split == 'test':
        paths.append(data / 'interactions_validation.csv.gz')
    paths.append(data / f'{split}_with_history.csv.gz')
    return paths
