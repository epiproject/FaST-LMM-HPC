#!/usr/bin/env python2

import time
from fastlmm.association import epistasis
from pysnptools.snpreader import Bed
from fastlmm.util.SamplePi import *
import pandas as pd

procs  = int(sys.argv[1])
data   = sys.argv[2]
pheno  = sys.argv[3]

runner = LocalMultiProc(procs, mkl_num_threads=1)

frame = epistasis(data, pheno, data, runner=runner)

print(frame)
