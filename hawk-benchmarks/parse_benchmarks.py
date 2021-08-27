#!/usr/bin/env python3
import csv
import re
import sys


def intervals(filename):
    with open(filename, 'r') as file:
        r = re.compile('\[(.*),(.*)\] (.*)')
        for line in file:
            m = r.match(line.strip())
            if m:
                try:
                    lb, ub, count = map(float, m.groups())
                except ValueError:
                    continue
                yield (f'{lb:.2f}', f'{ub:.2f}', count)


writer = csv.writer(sys.stdout, dialect='excel')

files = sys.argv[1:]
writer.writerow(['lb', 'ub'] + files)

for results in zip(*map(intervals, files)):
    # print(list(list(results)[0]))
    lb = results[0][0]
    ub = results[0][1]
    for (l, u, count) in results:
        assert l == lb
        assert u == ub
    writer.writerow([lb, ub] + [count for _, _, count in results])
