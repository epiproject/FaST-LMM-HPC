'''
Runs one part of a distributable job locally. The last part will return the jobs value. The other parts return 'None'

See SamplePi.py for examples.
'''
from fastlmm.util.runner import *
import os, sys
import logging
import fastlmm.util.util as util
import cPickle as pickle

class LocalInParts: # implements IRunner

    def __init__(self, taskindex, taskcount, mkl_num_threads, result_file=None, run_dir=".", logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
        logger.addHandler(logging_handler)

        self.run_dir = run_dir
        self.result_file = os.path.join(run_dir,result_file) if result_file else None
        self.taskindex = taskindex
        self.taskcount = taskcount
        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


    def run(self, distributable):
        tempdir = os.path.join(self.run_dir,distributable.tempdirectory)
        if self.taskindex != self.taskcount:
            JustCheckExists().input(distributable)
            return run_one_task(distributable, self.taskindex, self.taskcount, tempdir)
        else:
            result = run_one_task(distributable, self.taskindex, self.taskcount, tempdir)
            if self.result_file is not None:
                util.create_directory_if_necessary(self.result_file)
                with open(self.result_file, mode='wb') as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

            return result
