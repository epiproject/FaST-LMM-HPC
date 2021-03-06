from fastlmmhpc.util.runner import *
import logging
import fastlmmhpc.pyplink.plink as plink
import pysnptools.util as pstutil
import pysnptools.util.pheno as pstpheno
import fastlmmhpc.util.util as flutil
import numpy as np
from fastlmmhpc.inference import LMM
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmmhpc.util.pickle_io import load, save
import time
import pandas as pd
import fastlmmhpc.util.util as util
import cPickle as pickle
import sys

from multiprocessing import Lock, Manager, Queue

import threading
from fastlmmhpc.association import global_vars

from copy import copy, deepcopy


#----------------------------------------------------

####################### GPU #########################

#import numpy as np
#import skcuda.linalg as culinalg
#import skcuda.misc as cumisc
#import scikits.cuda.cublas as cublas
#import pycuda.tools as pytools

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

culinalg = None

####################### GPU #########################

#import gnumpy as gpu

##################################################################################################################################
#def my_doMainWorkForOneIndex(distributable, taskAndWorkcount, taskindex, workdirectory):    

def my_run_one_task(distributable, taskindex, taskcount, workdirectory):
    if not 0 < taskcount: raise Exception("Expect taskcount to be positive")
    if not (0 <= taskindex and taskindex < taskcount+1) :raise Exception("Expect taskindex to be between 0 (inclusive) and taskcount (exclusive)")

    global culinalg
    import pycuda.autoinit    
    import skcuda.linalg as culinalg

    culinalg.init()
    distributable.gpu_free.value = drv.mem_get_info()[0]

    shaped_distributable = shape_to_desired_workcount(distributable, taskcount)

    if shaped_distributable.work_count != taskcount : raise Exception("Assert: expect workcount == taskcount")

    util.create_directory_if_necessary(workdirectory, isfile=False)
    
    task_file_name = create_task_file_name(workdirectory, taskindex, taskcount)
    workDone = False
    with open(task_file_name, mode='wb') as f:
        is_first_and_only = True
        for work in shaped_distributable.work_sequence_range(taskindex, taskindex+1):
            assert is_first_and_only, "real assert"
            is_first_and_only = False
            result = run_all_in_memory(work)
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL) # save a result to the temp results file
    return None



def my_worker(distributablep_filename, runner_string, l, c, sync, g, tsk_id, tskcount, q, distributable):

    distributable.lock = l
    distributable.cond = c
    distributable.sync = sync
    distributable.myproc   = tsk_id
    distributable.numprocs = tskcount
    distributable.queue = q
    distributable.gpu_free = g


    return my_run_one_task(distributable, tsk_id - 1, tskcount, distributable.tempdirectory)


def epistasis(test_snps,pheno,G0, G1=None, mixing=0.0, covar=None,output_file_name=None,sid_list_0=None,sid_list_1=None,
              log_delta=None, min_log_delta=-5, max_log_delta=10, 
              cache_file = None,
              runner=None,
              mode = 0,
              max_gpu_procs = 1,
              split_matrix  = 100):

    #if runner is None:
        #runner = Local()

    #epistasis = _Epistasis(test_snps,pheno,G0, G1, mixing, covar,sid_list_0,sid_list_1, log_delta, min_log_delta, max_log_delta, output_file_name, cache_file)
    #epistasis.fill_in_cache_file()

    #drv.init()
    #ngpus = drv.Device.count()
    #print "NGPUS: ", ngpus


    if runner is None:
        runner = Local()

    epistasis = _Epistasis(test_snps,pheno,G0, G1, mixing, covar,sid_list_0,sid_list_1, log_delta, min_log_delta, max_log_delta, output_file_name, cache_file)

    epistasis.mode = mode
    epistasis.max_gpu_procs = max_gpu_procs
    epistasis.split_matrix  = split_matrix

    epistasis.fill_in_cache_file()
    
    print ("START Runner!!...")
    t = time.time()
    #result = runner.run(epistasis)
    #print "FINISHED Runner in %0.4f (s)!!" % (time.time() - t)

    #return result
    if mode == 0:
        ###########################################################################################################
        localpath = os.environ["PATH"]
        localwd = os.getcwd()
        
        import datetime
        now = datetime.datetime.now()
        run_dir_rel = os.path.join("runs",util.datestamp(appendrandom=True))
        run_dir_abs = os.path.join(localwd,run_dir_rel)
        util.create_directory_if_necessary(run_dir_rel, isfile=False)
        
        distributablep_filename = os.path.join(run_dir_rel, "distributable.p")
        with open(distributablep_filename, mode='wb') as f:
            pickle.dump(epistasis, f, pickle.HIGHEST_PROTOCOL)
    
        runner_cmd = "LocalInParts({0},{1},mkl_num_threads={2})".format("{0}", runner.taskcount, runner.mkl_num_threads)

        if not os.path.exists(distributablep_filename): raise Exception(distributablep_filename + " does not exist")
    
        lock      = Lock()
        cond      = multiprocessing.Condition()
        manager   = multiprocessing.Manager()
        sync      = manager.Value('i', 0)
        gpu_free  = manager.Value('i', 0)

        queue   = Queue()
        jobs = []

        for i in range(runner.taskcount):

            command_string = runner_cmd.format(i)
            p = multiprocessing.Process(target=my_worker, args=(distributablep_filename, command_string, lock, cond, sync, gpu_free, i + 1, runner.taskcount, queue, epistasis))
            jobs.append(p)
            p.start()
    
        for p in jobs:
            res = p.join()

        epistasis.last_pro = True
        result_sequence = work_sequence_from_disk(epistasis.tempdirectory, runner.taskcount)
        shaped_distributable = shape_to_desired_workcount(epistasis, runner.taskcount)
        result = shaped_distributable.reduce(result_sequence)

        JustCheckExists().output(epistasis)
        ###########################################################################################################
    else:
        ###########################################################################################################
        JustCheckExists().input(epistasis)
    
        localpath = os.environ["PATH"]
        localwd = os.getcwd()
        
        import datetime
        now = datetime.datetime.now()
        run_dir_rel = os.path.join("runs",util.datestamp(appendrandom=True))
        run_dir_abs = os.path.join(localwd,run_dir_rel)
        util.create_directory_if_necessary(run_dir_rel, isfile=False)
    
        distributablep_filename = os.path.join(run_dir_rel, "distributable.p")
        with open(distributablep_filename, mode='wb') as f:
            pickle.dump(epistasis, f, pickle.HIGHEST_PROTOCOL)
        
        distributable_py_file = os.path.join(os.path.dirname(__file__), "../util/", "distributable.py")
        if not os.path.exists(distributable_py_file): raise Exception("Expect file at " + distributable_py_file + ", but it doesn't exist.")
        command_format_string = sys.executable + " " + distributable_py_file + " " + distributablep_filename +" LocalInParts({0},{1},mkl_num_threads={2})".format("{0}", runner.taskcount, runner.mkl_num_threads)
        
        proc_list = []
        
        for taskindex in xrange(runner.taskcount):
            command_string = command_format_string.format(taskindex)
            proc = subprocess.Popen(command_string.split(" "), cwd=os.getcwd()) 
            proc_list.append(proc)

        for taskindex, proc in enumerate(proc_list):            
            rc = proc.wait()

        if not 0 == rc : raise Exception("Running python in python results in non-zero return code in task#{0}".format(taskindex))

        result = run_one_task(epistasis, runner.taskcount, runner.taskcount, epistasis.tempdirectory)

        JustCheckExists().output(epistasis)
        ###########################################################################################################

    print ("FINISH PROCESS IN: %0.4f (s)" % (time.time() - t))

    return result

##################################################################################################################################



def write(sid0_list, sid1_list, pvalue_list, output_file):
    """
    Given three arrays of the same length [as per the output of epistasis(...)], writes a header and the values to the given output file.
    """
    with open(output_file,"w") as out_fp:
        out_fp.write("{0}\t{1}\t{2}\n".format("sid0","sid1","pvalue"))
        for i in xrange(len(pvalue_list)):
            out_fp.write("{0}\t{1}\t{2}\n".format(sid0_list[i],sid1_list[i],pvalue_list[i]))


class _PackEpistasis(object) :
    def __init__(self, X, UX, UUX, sid0_list, sid1_list):
        self.X   = X
        self.UX  = UX
        self.UUX = UUX
        self.sid0_list = sid0_list
        self.sid1_list = sid1_list
        
# could this be written without the inside-out of IDistributable?
class _Epistasis(object) : #implements IDistributable

    def __init__(self,test_snps,pheno,G0, G1=None, mixing=0.0, covar=None,sid_list_0=None,sid_list_1=None,
                 log_delta=None, min_log_delta=-5, max_log_delta=10, output_file=None, cache_file=None):
        
        self.sync      = None
        self.lock      = None
        self.cond      = None
        self.myproc    = 0
        self.numprocs  = 0
        self.last_pro  = False
        self.mode      = 0
        self.queue     = None
        self.gpu_free  = 0
        self.max_gpu_procs = -1
        self.split_matrix  = -1

        self.test_snps = test_snps
        self.pheno = pheno
        self.output_file_or_none = output_file
        self.cache_file = cache_file
        self.covar = covar
        self.sid_list_0 = sid_list_0
        self.sid_list_1 = sid_list_1
        self.G0=G0
        self.G1_or_none=G1
        self.mixing=mixing
        self.external_log_delta=log_delta
        self.min_log_delta = min_log_delta
        self.max_log_delta = max_log_delta
        self._ran_once = False
        self._str = "{0}({1},{2},G0={6},G1={7},mixing={8},covar={3},output_file={12},sid_list_0={4},sid_list_1{5},log_delta={9},min_log_delta={10},max_log_delta={11},cache_file={13})".format(
            self.__class__.__name__, self.test_snps,self.pheno,self.covar,self.sid_list_0,self.sid_list_1,
                 self.G0, self.G1_or_none, self.mixing, self.external_log_delta, self.min_log_delta, self.max_log_delta, output_file, cache_file)
        
        #self.block_size = 10400        
        self.block_size = 6000

    def set_sid_sets(self):
        sid_set_0 = set(self.sid_list_0)
        self.intersect = sid_set_0.intersection(self.sid_list_1)
        self.just_sid_0 = sid_set_0.difference(self.intersect)
        self.just_sid_1 = self.intersect.symmetric_difference(self.sid_list_1)
        self._pair_count = len(self.just_sid_0)*len(self.intersect) + len(self.just_sid_0)*len(self.just_sid_1) + len(self.intersect)*len(self.just_sid_1) + len(self.intersect) * (len(self.intersect)-1)//2
        self.test_snps, self.pheno, self.covar, self.G0, self.G1_or_none = pstutil.intersect_apply([self.test_snps, self.pheno, self.covar, self.G0, self.G1_or_none]) #should put G0 and G1 first

    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = None

        if isinstance(self.test_snps, str):
            self.test_snps = Bed(self.test_snps)

        if isinstance(self.G0, str):
            self.G0 = Bed(self.G0)

        if isinstance(self.pheno, str):
            self.pheno = pstpheno.loadOnePhen(self.pheno,vectorize=True) #!! what about missing=-9?

        if self.covar is not None and isinstance(self.covar, str):
            self.covar = pstpheno.loadPhen(self.covar)#!! what about missing=-9?

        if self.G1_or_none is not None and isinstance(self.G1_or_none, str):
            self.G1_or_none = Bed(self.G1_or_none)

        if self.sid_list_0 is None:
            self.sid_list_0 = self.test_snps.sid

        if self.sid_list_1 is None:
            self.sid_list_1 = self.test_snps.sid

        self.set_sid_sets()

        #!!Should fix up to add only of no constant columns - will need to add a test case for this
        if self.covar is None:
            self.covar = np.ones((self.test_snps.iid_count, 1))
        else:
            self.covar = np.hstack((self.covar['vals'],np.ones((self.test_snps.iid_count, 1))))
        self.n_cov = self.covar.shape[1] 


        if self.output_file_or_none is None:
            self.__tempdirectory = ".working"
        else:
            self.__tempdirectory = self.output_file_or_none + ".working"

        self._ran_once = True
        

 #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        self._run_once()
        block_count = self.div_ceil(self._pair_count, self.block_size)
        return block_count



    def work_sequence(self):
        self._run_once()
        return self.work_sequence_range(0, self.work_count)

    '''
    def work_sequence_range(self, start, end):
        self._run_once()

        lmm = self.lmm_from_cache_file()
        lmm.sety(self.pheno['vals'])

        for sid0_list, sid1_list in self.pair_block_sequence_range(start, end):
            yield lambda lmm=lmm, sid0_list=sid0_list, sid1_list=sid1_list : self.do_work(lmm, sid0_list, sid1_list)
    '''
    
    def work_sequence_range(self, start, end):
        self._run_once()

        lmm = self.lmm_from_cache_file()
        lmm.sety(self.pheno['vals'])

        ##################################################################################################
        if self.mode == 1:
            for sid0_list, sid1_list in self.pair_block_sequence_range(start, end):
                yield lambda lmm=lmm, sid0_list=sid0_list, sid1_list=sid1_list : self.do_work(lmm, sid0_list, sid1_list)
            return
        ##################################################################################################
            

        if self.last_pro:
            for sid0_list, sid1_list in self.pair_block_sequence_range(start, end):
                yield lambda lmm=lmm, sid0_list=sid0_list, sid1_list=sid1_list : self.do_work(lmm, sid0_list, sid1_list)
            return


        #----------------------------- GPU Version ---------------------------------------#        
        #----------------------------- GPU Version ---------------------------------------#
        
        pack_list = []
        #self.lock.acquire()
        #print "PRocesss in GPU"
        #try:        
        
        #print "Free memory Before asign:", drv.mem_get_info()
        final_t = 0

        maxProducts = 1
        products = 0
        for sid0_list, sid1_list in self.pair_block_sequence_range(start, end):
            yield lambda lmm=lmm, sid0_list=sid0_list, sid1_list=sid1_list : self.do_work(lmm, sid0_list, sid1_list)            
            #products += 1
            #if products == maxProducts:
            #    global_vars.maxM = True

        
            #print "[%i]Primera Fase: %0.4f(s) Desglose [Funcion: %0.4f(s) + Otro: %0.4f(s)], Segunda Fase: %0.4f(s), Desglose GPU Funcion  [CPU Mult (%d): %0.4f(s) + COPY Fortran: %0.4f(s) + GPU Transfer: %0.4f(s) + GPU dot (%d | %d | %d): %0.4f(s) + GPU Free: %0.4f(s)]" % (self.myproc, global_vars.first_section, global_vars.call_func, global_vars.other_func, global_vars.second_section, global_vars.n_cpu_mult, global_vars.cpu_dot, global_vars.cpy_fortran, global_vars.gpu_transfer, global_vars.n_gpu_mult1, global_vars.n_gpu_mult2, global_vars.n_gpu_mult3, global_vars.gpu_dot, global_vars.gpu_free)

        #print "[Global Timing  :::  (Total = %0.2f s) => (1st Section = %0.2f s) + (2nd Section = %0.2f s)], [GPU Mult = %d, CPU Mult = %d]\n" % (global_vars.first_section + global_vars.second_section, global_vars.first_section, global_vars.second_section, global_vars.gpu_mult, global_vars.cpu_mult)

        #print "[Global Timing  :::  (Total = %0.2f s) => (1st Section = %0.2f s) + (2nd Section = %0.2f s)]\n[2 Section Timing  :::  (Total = %0.2f s) => (Select Columns = %0.2f s) + (Compute log-likelihood = %0.2f s) + (Select columns = %0.2f s) + (Compute log-likelihood = %0.2f s) + (Compute pValue = %0.2f s) + (Store Results = %0.2f s)]\n[Section nLLeval  :::  (nLL_p1 = %0.2f s) + (nLL_p2 = %0.2f) + (nLL_p3 = %0.2f s)]" % (global_vars.first_section + global_vars.second_section, global_vars.first_section, global_vars.second_section, global_vars.second_section, global_vars.snd_col1, global_vars.snd_log1, global_vars.snd_col2, global_vars.snd_log2, global_vars.snd_chi, global_vars.snd_res, global_vars.nLL_p1, global_vars.nLL_p2, global_vars.nLL_p3)

    def reduce(self, result_sequence):
        #doesn't need "run_once()"

        frame = pd.concat(result_sequence)
        frame.sort_values("PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if self.output_file_or_none is not None:
            frame.to_csv(self.output_file_or_none, sep="\t", index=False)

        return frame

        #!!Find a place to output info like this near the end of the run
        #logging.info("PhenotypeName\t{0}".format(pheno['header']))
        #logging.info("SampleSize\t{0}".format(test_snps.iid_count))
        #logging.info("SNPCount\t{0}".format(test_snps.sid_count))
        #logging.info("Runtime\t{0}".format(time.time()-t0))


    @property
    def tempdirectory(self):
        self._run_once()
        return self.__tempdirectory

    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        #Doesn't need run_once
        return self._str


    def copyinputs(self, copier):
        self._run_once()
        if isinstance(self.test_snps, str):
            copier.input(self.test_snps + ".bed")
            copier.input(self.test_snps + ".bim")
            copier.input(self.test_snps + ".fam")
        else:
            copier.input(self.test_snps)

        copier.input(self.pheno)
        copier.input(self.covar)

        if isinstance(self.G0, str):
            copier.input(self.G0 + ".bed")
            copier.input(self.G0 + ".bim")
            copier.input(self.G0 + ".fam")
        else:
            copier.input(self.G0)

        copier.input(self.G1_or_none)
        copier.input(self.cache_file)

    def copyoutputs(self,copier):
        #Doesn't need run_once
        copier.output(self.output_file_or_none)

 #end of IDistributable interface---------------------------------------

    @staticmethod
    def div_ceil(num, den): #!!move to utils?
        return -(-num//den) #The -/- trick makes it do ceiling instead of floor. "//" will do integer division even in the future and on floats.
    
    def pair_block_sequence_range(self, block_start, block_end):
        self._run_once()
        assert 0 <= block_start and block_start <= block_end and block_end <= self.work_count, "real assert"

        block_index = block_start
        start = block_index * self.pair_count // self.work_count
        next_start = (block_index+1) * self.pair_count // self.work_count
        size_goal = next_start - start
        end = block_end * self.pair_count // self.work_count
        
       #print os.getpid(), ": Block_start: ", start, "Block end:", end
        
        sid0_list = []
        sid1_list = []
        for sid0, sid1 in self.pair_sequence_range(start,end):
            sid0_list.append(sid0)
            sid1_list.append(sid1)
            
            if len(sid0_list) == size_goal:
                yield sid0_list, sid1_list
                block_index += 1
                if block_index == block_end:
                    return
                sid0_list = []
                sid1_list = []
                start = next_start
                next_start = (block_index+1) * self.pair_count // self.work_count
                size_goal = next_start - start
        assert len(sid0_list) == 0, "real assert"

    #If start == end, then returns without yielding anything 
    def pair_sequence_range(self,start,end):
        self._run_once()
        assert 0 <= start and start <= end and end <= self._pair_count, "real assert"

        i = start
        for sid0, sid1 in self.pair_sequence_with_start(start):
            yield sid0, sid1
            i = i + 1
            if i == end:
                break
        assert i == end, "Not enough items found. Didn't get to the end"


    def pair_sequence_with_start(self,start):
        self._run_once()

        skip_ref = [start]

                
        just_sid_0_list = list(self.just_sid_0)
        just_sid_1_list = list(self.just_sid_1)
        intersect_list = list(self.intersect)

        for sid0, sid1 in self.combo_distinct(just_sid_0_list, intersect_list, skip_ref):
            yield sid0, sid1
            
        for sid0, sid1 in self.combo_distinct(just_sid_0_list, just_sid_1_list, skip_ref):
            yield sid0, sid1
            
        for sid0, sid1 in self.combo_distinct(intersect_list, just_sid_1_list, skip_ref):
            yield sid0, sid1
            
        for sid0, sid1 in self.combo_same(intersect_list, skip_ref):
            yield sid0, sid1
            
        assert skip_ref[0] == 0, "real assert"


    def combo_distinct(self, distinct__list0, distinct__list1, skip_ref):
        row_count = len(distinct__list0)
        col_count = len(distinct__list1)

        if skip_ref[0] >= row_count * col_count:
            skip_ref[0] = skip_ref[0] - row_count * col_count
            assert skip_ref[0] >=0, "real assert"
            return

        row_start = skip_ref[0] // col_count
        skip_ref[0] = skip_ref[0] - row_start * col_count
        assert skip_ref[0] >=0, "real assert"

        for row_index in xrange(row_start, row_count):
            sid0 = distinct__list0[row_index]
            if row_index == row_start:
                col_start = skip_ref[0]
                skip_ref[0] = 0
            else:
                col_start = 0
            for col_index in xrange(col_start, col_count):
                sid1 = distinct__list1[col_index]
                yield sid0, sid1

    def combo_same(self, list, skip_ref):
        count = len(list)
        full_size = count * (count + 1) // 2
        if skip_ref[0] >= full_size:
            skip_ref[0] = skip_ref[0] - full_size
            assert skip_ref[0] >=0, "real assert"
            return

        row_start = int((-1 + 2*count - np.sqrt(1 - 4*count + 4*count**2 - 8*skip_ref[0]))/2)
        skip_ref[0] = skip_ref[0] - (count*row_start - (row_start*(1 + row_start))//2)
        assert skip_ref[0] >=0, "real assert"

        for row_index in xrange(row_start, count):
            sid0 = list[row_index]
            if row_index == row_start:
                col_start = skip_ref[0]
                skip_ref[0] = 0
            else:
                col_start = 0
            for col_index in xrange(col_start + 1 + row_index, count):
                sid1 = list[col_index]
                assert sid0 is not sid1, "real assert"
                yield sid0, sid1



    @property
    def pair_count(self):
        self._run_once()
        return self._pair_count

    def lmm_from_cache_file(self):
        logging.info("Loading precomputation from {0}".format(self.cache_file))
        lmm = LMM()
        with np.load(self.cache_file) as data:
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
        return lmm

    def fill_in_cache_file(self):
        self._run_once()

        logging.info("filling in the cache_file and log_delta, as needed")

        if self.G1_or_none is None:
            self.G1val_or_none = None
        else:
            self.G1val_or_none = self.G1_or_none.read().val

            # The S and U are always cached, in case they are needed for the cluster or for multi-threaded runs
        if self.cache_file is None:
            self.cache_file = os.path.join(self.__tempdirectory, "cache_file.npz")
            if os.path.exists(self.cache_file): # If there is already a cache file in the temp directory, it must be removed because it might be out-of-date
                os.remove(self.cache_file)

        lmm = None
        if not os.path.exists(self.cache_file):
            logging.info("Precomputing eigen")
            lmm = LMM()

            G0_standardized = self.G0.read().standardize()
            
            lmm.setG(G0_standardized.val, self.G1val_or_none, a2=self.mixing)
            
            logging.info("Saving precomputation to {0}".format(self.cache_file))
            util.create_directory_if_necessary(self.cache_file)
            np.savez(self.cache_file, lmm.U,lmm.S) #using np.savez instead of pickle because it seems to be faster to read and write
            
        if self.external_log_delta is None:
            if lmm is None:
                lmm = self.lmm_from_cache_file()
            
            logging.info("searching for internal delta")
            lmm.setX(self.covar)
            lmm.sety(self.pheno['vals'])
            #log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count
            result = lmm.find_log_delta(REML=False, sid_count=self.G0.sid_count, min_log_delta=self.min_log_delta, max_log_delta=self.max_log_delta  ) #!!what about findA2H2? minH2=0.00001
            self.external_log_delta = result['log_delta']

        self.internal_delta = np.exp(self.external_log_delta) * self.G0.sid_count
        logging.info("internal_delta={0}".format(self.internal_delta))
        logging.info("external_log_delta={0}".format(self.external_log_delta))


    def matrix_mult_cpu_gpu(self, proc_id, lock, max_gpu_procs, gpu_sync, split_col, X, Y, k, N):

        #######################################################################################
        #######################################################################################
        '''
        XT = X.T
        XF = np.copy(XT, "F")
        
        X_gpu     = gpuarray.to_gpu(np.float64(XF))
        Y_gpu     = gpuarray.to_gpu(np.float64(Y))
        
        R_gpu     = culinalg.dot(X_gpu, Y_gpu)
        R         = R_gpu.get()

        if (k<N):
            U_gpu   = gpuarray.to_gpu(X)
            UUX_gpu = culinalg.dot(U_gpu, R_gpu)
            UUX     = Y - UUX_gpu.get()
                    
            U_gpu.gpudata.free()
            UUX_gpu.gpudata.free()
            R_gpu.gpudata.free()

            del R_gpu
            del U_gpu
            del UUX_gpu
            
        else:
            UUX = None

        X_gpu.gpudata.free()
        Y_gpu.gpudata.free()
        
        del X_gpu
        del Y_gpu
        
        return R, UUX
        '''
        #######################################################################################
        #######################################################################################

        dot_timer = 0
        transf    = 0
        
        XT = X.T

        R = None
        UUX = None
        
        wait   = True
        noGPU  = False
        allGPU = False
        
        col_p  = 0

        while wait:
            lock.acquire()
            if gpu_sync.value < max_gpu_procs:
                gpu_sync.value += 1
                wait = False
            lock.release()

            if wait:
                t = time.time()
                ##############################################################
                cols_list = []
                
                col_limit = col_p + split_col
                if col_limit  >= Y.shape[1]:
                    col_limit = Y.shape[1]
                    noGPU = True
                
                for c in range(col_p, col_limit):
                    cols_list.append(c)

                t1 = time.time()
                YM = Y[:, cols_list]
                splitR = XT.dot(YM)
                
                if R is None:
                    R = splitR
                else:
                    R = np.concatenate((R, splitR), axis=1)
                
                col_p = col_limit

                tt = time.time()        
                global_vars.cpu_dot += (tt - t)
                global_vars.n_cpu_mult += 1

                #print "%d CPU: %dx%d . %dx%d = %dx%d, TIME=%0.4f(s)" % (proc_id, len(XT), len(XT[0]), len(YM), len(YM[0]), len(splitR), len(splitR[0]), tt - t1)

                if noGPU:
                    break
                ##############################################################

        
        if not noGPU:
            if R is None:
                t = time.time()
                XF = np.copy(XT, "F")
                global_vars.cpy_fortran += (time.time() - t)
                
                t = time.time()
                X_gpu     = gpuarray.to_gpu(np.float64(XF))
                Y_gpu     = gpuarray.to_gpu(np.float64(Y))
                global_vars.gpu_transfer += (time.time() - t)
                
                t = time.time()
                R_gpu = culinalg.dot(X_gpu, Y_gpu)

                R = R_gpu.get()
                global_vars.gpu_dot += (time.time() - t)
                global_vars.n_gpu_mult1 += 1
                #print "%d GPU 1: %dx%d . %dx%d = %dx%d, TIME=%0.4f(s)" % (proc_id, len(X_gpu), len(X_gpu[0]), len(Y_gpu), len(Y_gpu[0]), len(R_gpu), len(R_gpu[0]), time.time() - t)

            else:
                t = time.time()
                XF = np.copy(XT, 'F')
                global_vars.cpy_fortran += (time.time() - t)

                t = time.time()
                cols_list = []
                for c in range(col_p, Y.shape[1]):
                    cols_list.append(c)

                X_gpu  = gpuarray.to_gpu(np.float64(XF))
                Y_gpu  = gpuarray.to_gpu(np.float64(Y[:, cols_list]))
                global_vars.gpu_transfer += (time.time() - t)
                
                t = time.time()
                R_gpu     = culinalg.dot(X_gpu, Y_gpu)
                splitR = R_gpu.get()
                R = np.concatenate((R, splitR), axis=1)
                global_vars.gpu_dot += (time.time() - t) 
                global_vars.n_gpu_mult2 += 1

                #print "%d GPU 2: %dx%d . %dx%d = %dx%d, TIME=%0.4f(s)" % (proc_id, len(X_gpu), len(X_gpu[0]), len(Y_gpu), len(Y_gpu[0]), len(R_gpu), len(R_gpu[0]), time.time() - t)

                t = time.time()
                R_gpu.gpudata.free()
                del R_gpu
                global_vars.gpu_free += (time.time() - t)

                if (k<N):
                    t = time.time()
                    RF = np.copy(R, 'F')
                    global_vars.cpy_fortran += (time.time() - t)
                    
                    t = time.time()
                    R_gpu = gpuarray.to_gpu(RF)
                    global_vars.gpu_transfer += (time.time() - t)
                    
                    
            if (k<N):
                t = time.time()
                U_gpu   = gpuarray.to_gpu(X)
                global_vars.gpu_transfer += (time.time() - t)

                t = time.time()
                UUX_gpu = culinalg.dot(U_gpu, R_gpu)
                UUX     = Y - UUX_gpu.get()
                global_vars.gpu_dot += (time.time() - t)
                global_vars.n_gpu_mult3 += 1
                #print "%d GPU 3: %dx%d . %dx%d = %dx%d, TIME=%0.4f(s)" % (proc_id, len(U_gpu), len(U_gpu[0]), len(R_gpu), len(R_gpu[0]), len(UUX_gpu), len(UUX_gpu[0]), time.time() - t)

                t = time.time()
                U_gpu.gpudata.free()
                UUX_gpu.gpudata.free()
                R_gpu.gpudata.free()

                del R_gpu
                del U_gpu
                del UUX_gpu
                global_vars.gpu_free += (time.time() - t)

            else:
                UUX = None
            
                
            t = time.time()
            X_gpu.gpudata.free()
            Y_gpu.gpudata.free()
                                
            del X_gpu
            del Y_gpu
            global_vars.gpu_free += (time.time() - t)

            lock.acquire()
            gpu_sync.value -= 1
            lock.release()

            return R, UUX
            
        else:
            if (k<N):
                #print "ALL IN CPU"
                t = time.time()
                UUX = Y - X.dot(R)
                global_vars.cpu_dot += (time.time() - t)
                global_vars.n_cpu_mult += 1
            else:
                UUX = None

        return R, UUX



    def do_work(self, lmm, sid0_list, sid1_list):
        
        '''
        if global_vars.maxM:
            dataframe = pd.DataFrame(
                index=np.arange(len(sid0_list)),
                columns=('SNP0', 'Chr0', 'GenDist0', 'ChrPos0', 'SNP1', 'Chr1', 'GenDist1', 'ChrPos1', 'PValue', 'NullLogLike', 'AltLogLike')
                )
            
            #!!Is this the only way to set types in a dataframe?
            dataframe['Chr0'] = dataframe['Chr0'].astype(np.float)
            dataframe['GenDist0'] = dataframe['GenDist0'].astype(np.float)
            dataframe['ChrPos0'] = dataframe['ChrPos0'].astype(np.float)
            dataframe['Chr1'] = dataframe['Chr1'].astype(np.float)
            dataframe['GenDist1'] = dataframe['GenDist1'].astype(np.float)
            dataframe['ChrPos1'] = dataframe['ChrPos1'].astype(np.float)
            dataframe['PValue'] = dataframe['PValue'].astype(np.float)
            dataframe['NullLogLike'] = dataframe['NullLogLike'].astype(np.float)
            dataframe['AltLogLike'] = dataframe['AltLogLike'].astype(np.float)
            
            return dataframe
        '''
        
        #print "START DO WORK!!!"
        #import skcuda.linalg as culinalg

        t_f = time.time()

        t_p = time.time()
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
        
        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()
        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)
        
        products = snps_read.val[:,sid0_index_list] * snps_read.val[:,sid1_index_list] # in the products matrix, each column i is the elementwise product of sid i in each list
        X = np.hstack((self.covar, snps_read.val, products))
        
        k = lmm.S.shape[0]
        N = X.shape[0]
        global_vars.other_func += (time.time() - t_p)
        
        #UX = lmm.U.T.dot(X)
        #print UX
        
        ##############################################################
        ##                     Split MATRIX Dot                     ##
        ##############################################################
        t_p = time.time()

        ###--------------------------- CUDA Version -------------------------##        
        gpu = False


        #while not gpu:
        self.lock.acquire()
        
        if (k<N):
            gpu_memory_need = 3*X.nbytes + lmm.U.T.nbytes + lmm.U.nbytes      
        else:
            gpu_memory_need = 2*X.nbytes + lmm.U.T.nbytes

            #print "%d < %d" % (gpu_memory_need, self.gpu_free.value)

        if gpu_memory_need < self.gpu_free.value:
            self.gpu_free.value -= gpu_memory_need
            gpu = True
                #print "GPU FREE"

        self.lock.release()
        
        #print "START GPU"
        #return 
        
        #gpu = False

        if gpu:
            #print "[%d] GPU RUN" % (self.myproc)
            U_T = np.copy(lmm.U.T, "F")

            X_gpu   = gpuarray.to_gpu(X)
            U_T_gpu = gpuarray.to_gpu(U_T)
            UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
            UX      = UX_gpu.get()

            if (k<N):
                #UUX = X - lmm.U.dot(UX)
                U_gpu   = gpuarray.to_gpu(lmm.U)
                UUX_gpu = culinalg.dot(U_gpu, UX_gpu)
                UUX     = X - UUX_gpu.get()

                U_gpu.gpudata.free()
                UUX_gpu.gpudata.free()
                
                del U_gpu
                del UUX_gpu

            else:
                UUX = None
        
            X_gpu.gpudata.free()
            U_T_gpu.gpudata.free()
            UX_gpu.gpudata.free()
            
            del X_gpu
            del U_T_gpu
            del UX_gpu     
            
            self.lock.acquire()
            self.gpu_free.value += gpu_memory_need
            self.lock.release()
            
        else:
            UX = lmm.U.T.dot(X)

            if (k<N):
                UUX = X - lmm.U.dot(UX)
            else:
                UUX = None

        ###--------------------------- CUDA Version -------------------------##

        #UX, UUX = self.matrix_mult_cpu_gpu(self.myproc, self.lock, self.max_gpu_procs, self.gpu_free, self.split_matrix, lmm.U, X, k, N)

        global_vars.call_func += (time.time() - t_p)
        #############################################################
        #############################################################
        #############################################################
        
        global_vars.first_section += (time.time() - t_f)

        dict_list = []
        
        t_second = time.time()
        
        for pair_index, sid0 in enumerate(sid0_list):
            sid1 = sid1_list[pair_index]
            sid0_index = sid0_index_list[pair_index]
            sid1_index = sid1_index_list[pair_index]
            
            index_list = np.array([pair_index]) #index to product
            index_list = index_list + len(sid_union_index_list) #Shift by the number of snps in the union
            index_list = np.hstack((np.array([sid0_index,sid1_index]),index_list)) # index to sid0 and sid1
            index_list = index_list + self.covar.shape[1] #Shift by the number of values in the covar
            index_list = np.hstack((np.arange(self.covar.shape[1]),index_list)) #indexes of the covar
            
            index_list_less_product = index_list[:-1] #index to everything but the product

            #Alt -- now with the product feature
            lmm.X = X[:,index_list]
            lmm.UX = UX[:,index_list]
            if (k<N):
                lmm.UUX = UUX[:,index_list]
            else:
                lmm.UUX = None
            res_alt = lmm.nLLeval(delta=self.internal_delta, REML=False, useMemorizedLogdetK=True)
            ll_alt = -res_alt["nLL"]
            
            #Null -- the two additive SNPs
            #lmm.X = X[:,index_list_less_product]
            #lmm.UX = UX[:,index_list_less_product]
            lmm.X = lmm.X[:,:-1]
            lmm.UX = lmm.UX[:,:-1]
            if (k<N):
                lmm.UUX = lmm.UUX[:,:-1]
                #lmm.UUX = UUX[:,index_list_less_product]
            else:
                lmm.UUX = None
            res_null = lmm.nLLeval(delta=self.internal_delta, REML=False, useMemorizedLogdetK=True)
            ll_null = -res_null["nLL"]
            
            
            test_statistic = ll_alt - ll_null
            degrees_of_freedom = 1
            pvalue = stats.chi2.sf(2.0 * test_statistic, degrees_of_freedom)
            logging.debug("<{0},{1}>, null={2}, alt={3}, pvalue={4}".format(sid0,sid1,ll_null,ll_alt,pvalue))
            
            '''
            dataframe.iloc[pair_index] = [
                 sid0, snps_read.pos[sid0_index,0],  snps_read.pos[sid0_index,1], snps_read.pos[sid0_index,2],
                 sid1, snps_read.pos[sid1_index,0],  snps_read.pos[sid1_index,1], snps_read.pos[sid1_index,2],
                 pvalue, ll_null, ll_alt]
            '''
            
            ##################################################################################################################################            
            #x = time.time()
            dict_frame = {}            
            dict_frame['SNP0']        = sid0
            dict_frame['Chr0']        = snps_read.pos[sid0_index,0]
            dict_frame['GenDist0']    = snps_read.pos[sid0_index,1]
            dict_frame['ChrPos0']     = snps_read.pos[sid0_index,2]
            dict_frame['SNP1']        = sid1
            dict_frame['Chr1']        = snps_read.pos[sid1_index,0]
            dict_frame['GenDist1']    = snps_read.pos[sid1_index,1]
            dict_frame['ChrPos1']     = snps_read.pos[sid1_index,2]
            dict_frame['PValue']      = pvalue
            dict_frame['NullLogLike'] = ll_null
            dict_frame['AltLogLike']  = ll_alt      
            dict_list.append(dict_frame)            
            #tot_iloc += (time.time() - x)
            ##################################################################################################################################
            
        #print "Timing acum in %d (%0.4f (s))" % (len(sid0_list) * 2, global_vars.log_time)
        
        dataframe = pd.DataFrame(dict_list)    
        global_vars.second_section += (time.time() - t_second)

        return dataframe


        #if (k<N):
        #    UUX = X - lmm.U.dot(UX)
        #else:
        #    UUX = None
        
        ##############################################################
        ##                     Split MATRIX Dot                     ##
        ##############################################################

        #if (k<N):
        #    UUX = X - self.matrix_mult_cpu_gpu(self.myproc, self.lock, 1, self.gpu_free, 500, lmm.U, UX, False)
            #UUX = X - lmm.U.dot(UX)
        #else:
        #    UUX = None
        
        #print R
        #self.lock.acquire()
        #print UX
        #print R
        #print "-------------------------------------------------------------"
        #self.lock.release()

        #for i in range(0, 1499):
        #    UX_c = lmm.U.T.dot(X[:,i])
        #    for p in range(0, len(UX[:,i])):
        #        val_1 = round(UX[:,i][p], 12)
        #        val_2 = round(UX_c[p], 12)
        #        val_3 = round(UX_g[:,i][p], 12)
        #        if (val_1 != val_2) or (val_1 != val_3):
        #            print val_1," vs ",val_2," vs ",val_3

        #print lmm.U.T[:,[0,1,2,3,5,6,7,8,9]][0:9]
        #print "-----------------------------------------------"
        #print X[:,[0,1,2,3,5,6,7,8,9]][0:9]
        #print "-----------------------------------------------"

        #print "U.T: ", lmm.U.T[:,0:4800][0:1].shape
        #print "X: ", X[0:4800].shape
        
        #U_T = np.copy(lmm.U.T, "F")
        #X_gpu   = gpuarray.to_gpu(np.float64(X))
        #U_T_gpu = gpuarray.to_gpu(np.float64(U_T))
        #UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
        #UX_g    = UX_gpu.get()

        '''
        UX = lmm.U.T.dot(X)
        if (k<N):
            UUX = X - lmm.U.dot(UX)
        else:
            UUX = None
        '''
        
        #UX_c = lmm.U.T.dot(X[:,0])
        
        #print UX[:,0][0], " vs ",UX_c[0]," vs ",UX_g[:,0][0]
        
        #acum = 0.0
        #for p in range(0,len(lmm.U.T[1])):
        #    acum += (lmm.U.T[0][p] * X[:,0][p])
        
            
            
        #for item in lmm.U.T[0]

        #print lmm.U.T.shape
        #print X.shape
        
        #UT = lmm.U.T[:,0:5][0:1]
        #X  = X[:,0][0:5]

        #print UT.shape
        #print UT
        #print "-----------------------------------------------------"
        #print X.shape
        #print X
        #print "-----------------------------------------------------"
        
        #UX = lmm.U.T.dot(X)
        
        #for r in lmm.U:
        #    for item in r:
        #        sys.stderr.write("%f " % item)
        #    sys.stderr.write("\n")

        '''
        print "UT:", lmm.U.T.shape
        print "X:", X.shape
        
        UX = lmm.U.T[:,0:4804].dot(X[0:4804])
        print "0:",UX[0][0], ",10:", UX[10][0], ",1000: ", UX[100][0], ",4802: ", UX[499][0]
        print "---------------------------------------"
        UX = lmm.U.T.dot(X)
        print "0:",UX[0][0], ",10:", UX[10][0], ",1000: ", UX[100][0], ",4802: ", UX[499][0]
        print "---------------------------------------"
        
        #U_TF     = np.copy(lmm.U, 'F')
        #XF       = np.copy(X)
        #print "UT:", U_TF.shape
        #print "X:", XF.shape
        #

        #UX = U_TF.dot(XF)

        UX = lmm.U.T.dot(X[:,0])
        #UX = lmm.U.T.dot(X[:,0])
        print "0:",UX[0], ",10:", UX[10], ",1000: ", UX[100], ",4802: ", UX[499]
        
        print "================================================================="
        
        X_gpu   = gpuarray.to_gpu(np.float64(X))
        U_T_gpu = gpuarray.to_gpu(np.float64(U_TF))
        
        #X_gpu   = gpuarray.to_gpu(np.float64(X[:,[0]][0:4804]))
        #U_T_gpu = gpuarray.to_gpu(np.float64(U_TF[:,0:4804]))

        print X_gpu.shape
        print U_T_gpu.shape

        UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
        UX      = UX_gpu.get()
        print UX[0]
        '''
        '''
        X2 = X[0:5][:,[0, 1, 2, 3, 4]]
        UT2 = lmm.U.T[0:5][:,[0, 1, 2, 3, 4]]

        print "UT:", UT2.shape
        print "X:", X2.shape

        UX = UT2[:,0:2].dot(X2[0:2])
        print UX

        print "---------------------------------------"
        
        UX = UT2[:,0:2].dot(X2[:,[0]][0:2])
        print UX[0]

        print "---------------------------------------"

        UX = UT2[:,0:2].dot(X2[:,[0,1]][0:2])
        print UX[0]
        '''

        #U_T     = np.copy(lmm.U.T, "F")
        #X_gpu   = gpuarray.to_gpu(X[0:4800])
        #U_T_gpu = gpuarray.to_gpu(U_T[:,0:4800])

        #print "U.T: ", U_T_gpu.shape
        #print "X: ", X_gpu.shape

        #UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
        #UX      = UX_gpu.get()
        
        #print U_T_gpu
        #print "-----------------------------------------------"
        #print X_gpu
        #print "-----------------------------------------------"
        #print UX
        #print "================================================================="
        
        '''
        UX = lmm.U.T.dot(X)
        if (k<N):
            UUX = X - lmm.U.dot(UX)
        else:
            UUX = None
        

        print UX[0][0], UX[0][1], UX[0][2]

        
        #print lmm.U.T
        #print X
        
        U_T     = np.copy(lmm.U.T, "F")
        X_gpu   = gpuarray.to_gpu(X)
        U_T_gpu = gpuarray.to_gpu(U_T)
        UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
        UX      = UX_gpu.get()

        print UX[0][0], UX[0][1], UX[0][2]
        print UX


        max_gpu_procs = 1
        split_col = 500
        #UX2 = self.matrix_mult_cpu_gpu(self.myproc, self.lock, max_gpu_procs, self.gpu_free, split_col, lmm.U.T, X)
        '''


    def do_work_first_normal(self, lmm, sid0_list, sid1_list):        
        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()
        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)
        
        products = snps_read.val[:,sid0_index_list] * snps_read.val[:,sid1_index_list] # in the products matrix, each column i is the elementwise product of sid i in each list
        X = np.hstack((self.covar, snps_read.val, products))

        t = time.time()
            #print "[%d] CPU RUN" % (self.myproc)
        UX = lmm.U.T.dot(X)
        ###--------------------------- CUDA Version -------------------------##

        #print UX.shape

        #tt = time.time()        
        #acum = tt - t
        #global_vars.first_section += acum

        #print "NEW Matrix Time First Matrix (%d x %d).dot(%d x %d) : Indiv=%0.4f (s): Tot=%0.4f (s)" % (len(lmm.U.T), len(lmm.U.T[0]), len(X), len(X[0]), tt - t, global_vars.first_section)

        k = lmm.S.shape[0]
        N = X.shape[0]
        
        if (k<N):
            UUX = X - lmm.U.dot(UX)
        else:
            UUX = None
        
        #pack = None
        pack = _PackEpistasis(X, UX, UUX, sid0_list, sid1_list)
                
        return pack
        '''

    def do_work_first(self, lmm, sid0_list, sid1_list):        
        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()
        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)
        
        products = snps_read.val[:,sid0_index_list] * snps_read.val[:,sid1_index_list] # in the products matrix, each column i is the elementwise product of sid i in each list
        X = np.hstack((self.covar, snps_read.val, products))

        t = time.time()
        #------------------------------ CONVERT FLOAT 32-------------------------------------#        
        #my_X = X.astype(np.float32)
        #my_U = lmm.U.astype(np.float32)
        #UX = my_U.T.dot(my_X)
        #------------------------------ CONVERT FLOAT 32-------------------------------------#  
        
        #run_in_gpu = False
        
        #self.lock.acquire()
        #if self.gpu_free.value:
            #run_in_gpu = True
            #self.gpu_free.value = False        
        #self.lock.release()

        #if run_in_gpu:
            #print "SEND TO GPU..."

        ###--------------------------- CUDA Version -------------------------##        
        gpu = False
        
        self.lock.acquire()
        gpu_memory_need = 2*X.nbytes + lmm.U.T.nbytes        
        if gpu_memory_need < self.gpu_free.value:
            self.gpu_free.value -= gpu_memory_need
            gpu = True
        self.lock.release()
        
        #print "[%d] Need: %d, Free: %d" % (self.myproc, gpu_memory_need, self.gpu_free.value)        
        
        if gpu:
            #print "[%d] GPU RUN" % (self.myproc)
            
            U_T = np.copy(lmm.U.T, "F")

            X_gpu   = gpuarray.to_gpu(X)
            U_gpu   = gpuarray.to_gpu(U_T)
            UX_gpu  = culinalg.dot(U_gpu, X_gpu)

            UX      = UX_gpu.get()
        
            X_gpu.gpudata.free()
            U_gpu.gpudata.free()
            UX_gpu.gpudata.free()
            
            del X_gpu
            del U_gpu
            del UX_gpu     
            
            self.lock.acquire()
            self.gpu_free.value += gpu_memory_need
            self.lock.release()
            
        else:
            #print "[%d] CPU RUN" % (self.myproc)
            UX = lmm.U.T.dot(X)
        ###--------------------------- CUDA Version -------------------------##

        #print UX.shape

        #tt = time.time()        
        #acum = tt - t
        #global_vars.first_section += acum

        #print "NEW Matrix Time First Matrix (%d x %d).dot(%d x %d) : Indiv=%0.4f (s): Tot=%0.4f (s)" % (len(lmm.U.T), len(lmm.U.T[0]), len(X), len(X[0]), tt - t, global_vars.first_section)

        k = lmm.S.shape[0]
        N = X.shape[0]
        
        if (k<N):
            UUX = X - lmm.U.dot(UX)
        else:
            UUX = None
        
        #pack = None
        pack = _PackEpistasis(X, UX, UUX, sid0_list, sid1_list)
                
        return pack
            
    def do_work_second(self, lmm, packEpi):
        
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))

        sid0_list = packEpi.sid0_list
        sid1_list = packEpi.sid1_list

        #calculate or pass by packEpi!!!???? Timing!!
        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()
        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)
        #--------------------------------------------
        
        UX = packEpi.UX
        UUX = packEpi.UUX
        X = packEpi.X
        
        k = lmm.S.shape[0]
        N = X.shape[0]


        dataframe = pd.DataFrame(
            index=np.arange(len(sid0_list)),
            columns=('SNP0', 'Chr0', 'GenDist0', 'ChrPos0', 'SNP1', 'Chr1', 'GenDist1', 'ChrPos1', 'PValue', 'NullLogLike', 'AltLogLike')
        )
        
        #!!Is this the only way to set types in a dataframe?
        dataframe['Chr0'] = dataframe['Chr0'].astype(np.float)
        dataframe['GenDist0'] = dataframe['GenDist0'].astype(np.float)
        dataframe['ChrPos0'] = dataframe['ChrPos0'].astype(np.float)
        dataframe['Chr1'] = dataframe['Chr1'].astype(np.float)
        dataframe['GenDist1'] = dataframe['GenDist1'].astype(np.float)
        dataframe['ChrPos1'] = dataframe['ChrPos1'].astype(np.float)
        dataframe['PValue'] = dataframe['PValue'].astype(np.float)
        dataframe['NullLogLike'] = dataframe['NullLogLike'].astype(np.float)
        dataframe['AltLogLike'] = dataframe['AltLogLike'].astype(np.float)

        
        dict_list = []
        
        for pair_index, sid0 in enumerate(sid0_list):
            sid1 = sid1_list[pair_index]
            sid0_index = sid0_index_list[pair_index]
            sid1_index = sid1_index_list[pair_index]

            index_list = np.array([pair_index]) #index to product
            index_list = index_list + len(sid_union_index_list) #Shift by the number of snps in the union
            index_list = np.hstack((np.array([sid0_index,sid1_index]),index_list)) # index to sid0 and sid1
            index_list = index_list + self.covar.shape[1] #Shift by the number of values in the covar
            index_list = np.hstack((np.arange(self.covar.shape[1]),index_list)) #indexes of the covar
            
            index_list_less_product = index_list[:-1] #index to everything but the product

            #Null -- the two additive SNPs
            lmm.X = X[:,index_list_less_product]
            lmm.UX = UX[:,index_list_less_product]
            if (k<N):
                lmm.UUX = UUX[:,index_list_less_product]
            else:
                lmm.UUX = None
            res_null = lmm.nLLeval(delta=self.internal_delta, REML=False)
            ll_null = -res_null["nLL"]

            #Alt -- now with the product feature
            lmm.X = X[:,index_list]
            lmm.UX = UX[:,index_list]
            if (k<N):
                lmm.UUX = UUX[:,index_list]
            else:
                lmm.UUX = None
            res_alt = lmm.nLLeval(delta=self.internal_delta, REML=False)
            ll_alt = -res_alt["nLL"]

            test_statistic = ll_alt - ll_null
            degrees_of_freedom = 1
            pvalue = stats.chi2.sf(2.0 * test_statistic, degrees_of_freedom)
            logging.debug("<{0},{1}>, null={2}, alt={3}, pvalue={4}".format(sid0,sid1,ll_null,ll_alt,pvalue))


            dataframe.iloc[pair_index] = [
                 sid0, snps_read.pos[sid0_index,0],  snps_read.pos[sid0_index,1], snps_read.pos[sid0_index,2],
                 sid1, snps_read.pos[sid1_index,0],  snps_read.pos[sid1_index,1], snps_read.pos[sid1_index,2],
                 pvalue, ll_null, ll_alt]

            
            ##################################################################################################################################            
            #x = time.time()
            dict_frame = {}            
            dict_frame['SNP0']        = sid0
            dict_frame['Chr0']        = snps_read.pos[sid0_index,0]
            dict_frame['GenDist0']    = snps_read.pos[sid0_index,1]
            dict_frame['ChrPos0']     = snps_read.pos[sid0_index,2]
            dict_frame['SNP1']        = sid1
            dict_frame['Chr1']        = snps_read.pos[sid1_index,0]
            dict_frame['GenDist1']    = snps_read.pos[sid1_index,1]
            dict_frame['ChrPos1']     = snps_read.pos[sid1_index,2]
            dict_frame['PValue']      = pvalue
            dict_frame['NullLogLike'] = ll_null
            dict_frame['AltLogLike']  = ll_alt      
            dict_list.append(dict_frame)            
            #tot_iloc += (time.time() - x)
            ##################################################################################################################################
            
        #print "Timing acum in %d (%0.4f (s))" % (len(sid0_list) * 2, global_vars.log_time)
        
        dataframe = pd.DataFrame(dict_list)    
        global_vars.log_time = 0

        return dataframe
        '''
    
if __name__ == "__main__":
    
    import doctest
    doctest.testmod()

    #print "done"

