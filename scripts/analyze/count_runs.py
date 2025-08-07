import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import argparse
import math
from scipy.stats import gmean

run_dir = Path(sys.argv[1])
p1 = run_dir

c = 0

task_cs = []

for d1 in os.listdir(p1):
    c_curr = 0

    p2 = p1 / d1 / "phases"
    for d2 in os.listdir(p2):
        p3 = p2 / d2 / "agents"
        for d3 in os.listdir(p3):
            p4 = p3 / d3
            l = len(os.listdir(p4))
            c += l
            c_curr += l
    
    task_cs.append(c_curr)

print(np.min(np.array(task_cs)))

print(c)
