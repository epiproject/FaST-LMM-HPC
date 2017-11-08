from fastlmm.util.runner import *
import logging
import fastlmm.pyplink.plink as plink
import pysnptools.util as pstutil
import pysnptools.util.pheno as pstpheno
import fastlmm.util.util as flutil
import numpy as np
from fastlmm.inference import LMM
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd
from mpi4py import MPI

from multiprocessing import Lock, Manager, Queue
from copy import copy, deepcopy
#----------------------------------------------------

####################### GPU #########################

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

culinalg = None

#import pycuda.autoinit
#import skcuda.linalg as culinalg
#import skcuda.linalg as culinalg
#import skcuda.misc as cumisc
#import scikits.cuda.cublas as cublas

####################### GPU #########################

from fastlmm.association import global_vars


def _worker(epistasis, distributablep_filename, runner_string, id_w, rank, work, q_items, queue, mutex, pairs_per_block, snps_list, lock, multi_gpu_free, tot_w):

    ####################### LOADING DATA EPISTASIS #########################

    with open(distributablep_filename, mode='rb') as f:
        try:
            distributable = pickle.load(f)
        except AttributeError, e:
            raise AttributeError("[Original message: '{0}'".format(e))
    exec("runner = " + runner_string)
    JustCheckExists().input(distributable)
    
    taskindex = runner.taskindex
    taskcount = runner.taskcount
    workdirectory = distributable.tempdirectory

    epistasis.multi_gpu_free = multi_gpu_free
    epistasis.lock = lock
    epistasis.id_w = id_w

    global culinalg
    #import pycuda.autoinit
    import skcuda.linalg as culinalg
    import skcuda

    #epistasis.gpu_free.value = drv.mem_get_info()[0]
    
    GPUNum = drv.Device.count()
    if GPUNum > 20:
        GPUNum = 20

    if GPUNum > id_w:
        device = id_w
    else:
        device = id_w % GPUNum
    
    #device = 0
    
    gpuDev = skcuda.misc.init_device(device)
    skcuda.misc.init_context(gpuDev)
    culinalg.init(allocator=drv.mem_alloc)
    
    epistasis.my_gpu = device
    epistasis.multi_gpu_free[device] = drv.mem_get_info()[0]

    print "START Process %d GPU[%d/%d] : Memory free %f" % (id_w, device, GPUNum, epistasis.multi_gpu_free[device])

    #print epistasis.gpu_free.value
    #return 0

    #if not 0 < taskcount: raise Exception("Expect taskcount to be positive")
    #if not (0 <= taskindex and taskindex < taskcount+1) :raise Exception("Expect taskindex to be between 0 (inclusive) and taskcount (exclusive)")    
    #shaped_distributable = shape_to_desired_workcount(distributable, taskcount)
    #if shaped_distributable.work_count != taskcount : raise Exception("Assert: expect workcount == taskcount")

    util.create_directory_if_necessary(workdirectory, isfile=False)
    task_file_name = create_task_file_name(workdirectory, taskindex, taskcount)
    f = open(task_file_name, "wb")
    
    #with open(task_file_name, mode='wb') as f:
    #is_first_and_only = True
    #for work in shaped_distributable.work_sequence_range(taskindex, taskindex+1):
    #assert is_first_and_only, "real assert"
    #is_first_and_only = False
            
    #result = run_all_in_memory(work)
    #pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    #==============================================#
    results_list = []
    #==============================================#
    epistasis._run_once()
    lmm = epistasis.lmm_from_cache_file()
    lmm.sety(epistasis.pheno['vals'])    
    ####################### LOADED DATA EPISTASIS #########################


    ######################### MPI WORKER SECTION ##########################
    while True:
        mutex.acquire()
        item = 0
        if not queue.empty():
            item = queue.get()
            q_items.value -= 1
        mutex.release()
        
        if item:
            #print "[%d]PROCESS ITEM" % (rank)
            ################# Work #####################
            snp0_list, snp1_list = making_pairs(item, pairs_per_block, snps_list)

            result = epistasis.do_work(lmm, snp1_list, snp0_list)
            results_list.append((taskindex, result))
            ############################################
        if not work.value and queue.empty():
            #print "[%d]BREAK LOOP" % (rank)
            break
    ######################### MPI WORKER SECTION ##########################

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print "@@rank\t%d\tthread\t%d\tFirst Section\t%0.2f\tSecond Section\t%0.2f" % (rank, id_w, global_vars.first_section, global_vars.second_section)
    #print "[%d]First Section %0.2f, Second Section %0.2f, Total %0.2f, By node: Otro: %0.2f, nLLeval 1: %0.2f, nLLeval 2: %0.2f,  chi2 : %0.2f, Dataframe: %0.2f" % (rank, global_vars.first_section, global_vars.second_section, global_vars.first_section + global_vars.second_section, global_vars.l_tot1, global_vars.l_tot2, global_vars.l_tot3, global_vars.l_tot4, global_vars.l_tot5)


    pickle.dump(results_list, f, pickle.HIGHEST_PROTOCOL)

    return


def making_pairs(pairs_range, pairs_per_block, snps_list):
    tot_snps = len(snps_list)
    row      = pairs_range[0]
    col      = pairs_range[1]

    snp0_list = []
    snp1_list = []
    
    start = col
    for p0 in range(row, tot_snps):
        for p1 in range(start + 1, tot_snps):
            snp0_list.append(snps_list[p0])
            snp1_list.append(snps_list[p1])
            if len(snp0_list) >= pairs_per_block:                
                break
        start = p0 + 1
        if len(snp0_list) >= pairs_per_block:                
            break

    return snp0_list, snp1_list


def epistasis(test_snps, pheno, G0, taskcount, 
              pairs_per_block=1000, G1=None, mixing=0.0, covar=None, output_file_name=None,
              sid_list_0=None,sid_list_1=None,
              log_delta=None, min_log_delta=-5, max_log_delta=10, 
              cache_file = None):

    t = time.time()
    runner = LocalMultiProc(taskcount, mkl_num_threads=1)    
    epistasis = _Epistasis(test_snps, pheno, G0, G1, mixing, covar, sid_list_0, sid_list_1, log_delta, min_log_delta, max_log_delta, output_file_name, cache_file)
    
    epistasis.fill_in_cache_file()
    
    print "====== FILL IN CAHE FILE FINISHED IN %lu ======" % (time.time() - t)
    
    snps_list = []
    fd = open(test_snps + ".bim", "r")
    for l in fd:
        spl = l.split("\t")
        snps_list.append(spl[1])

    #print "====== SNPS LOADED ======"
    #print snps_list
    #print "========================="

    ######################## EPISTASIS DATA PREPARATION #############################
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
    
    tunner = "LocalInParts({0},{1},mkl_num_threads={2})".format("{0}", runner.taskcount, runner.mkl_num_threads)

    if not os.path.exists(distributablep_filename): raise Exception(distributablep_filename + " does not exist") 
    #################################################################################

    ######### MAKING WORKERS PROCESSES ##########
    comm            = MPI.COMM_WORLD
    rank            = comm.Get_rank()
    mpi_procs       = comm.Get_size()

    jobs = []

    manager = multiprocessing.Manager()
    work    = manager.Value('i', 1)
    q_items = manager.Value('i', 0)
    queue   = manager.Queue()
    mutex   = multiprocessing.Lock()
    lock    = Lock()

    multi_gpu_free = manager.Array('f', range(20))
    
    for i in range(taskcount):
        command_string = tunner.format(i)

        #TODO: MEMORY USAGE DANGER! SNPS_LIST WILL BE COPY FOR EACH PROCESS
        p = multiprocessing.Process(target=_worker, args=(epistasis, distributablep_filename, command_string, i, rank, 
                                                          work, q_items, queue, mutex, pairs_per_block, snps_list, lock, multi_gpu_free, taskcount))
        jobs.append(p)
        p.start()
    
    while True:
        #print "[Worker %d] Send petition to Manager " % (rank)
        comm.send(rank, dest=mpi_procs - 1, tag=1)
        task = comm.recv(source=mpi_procs - 1, tag=2)
        if len(task):
            #print "[Worker %d] Work Recived: " % (rank), task
            mutex.acquire()
            queue.put(task)
            q_items.value += 1
            mutex.release()
        else:
            #print "[Worker %d] Not more work! Finish!??" % (rank)
            comm.send(-1, dest=mpi_procs - 1, tag=1)
            work.value = 0
            break

        #print "In <----- QUEUE ITEMS: %d, TASKCOUNT: %d" % (q_items.value, taskcount)
        while q_items.value == (taskcount + 2):
            time.sleep(1)
            #pass
        #print "Out <----- QUEUE ITEMS: %d, TASKCOUNT: %d" % (q_items.value, taskcount)

    for p in jobs:
        res = p.join()

    
    
    return
    #return result


def write(sid0_list, sid1_list, pvalue_list, output_file):
    """
    Given three arrays of the same length [as per the output of epistasis(...)], writes a header and the values to the given output file.
    """
    with open(output_file,"w") as out_fp:
        out_fp.write("{0}\t{1}\t{2}\n".format("sid0","sid1","pvalue"))
        for i in xrange(len(pvalue_list)):
            out_fp.write("{0}\t{1}\t{2}\n".format(sid0_list[i],sid1_list[i],pvalue_list[i]))


# could this be written without the inside-out of IDistributable?
class _Epistasis(object) : #implements IDistributable

    def __init__(self,test_snps,pheno,G0, G1=None, mixing=0.0, covar=None,sid_list_0=None,sid_list_1=None,
                 log_delta=None, min_log_delta=-5, max_log_delta=10, output_file=None, cache_file=None):
        self.lock      = None
        self.multi_gpu_free  = None
        self.my_gpu = 0
        self.id_w = 0
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
        self.block_size = 16000

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

        rank = str(MPI.COMM_WORLD.Get_rank())
        if self.output_file_or_none is None:
            self.__tempdirectory = ".working."+rank
        else:
            self.__tempdirectory = self.output_file_or_none + ".working."+rank

        self._ran_once = True
        

 #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        self._run_once()
        block_count = self.div_ceil(self._pair_count, self.block_size)
        return block_count



    def work_sequence(self):
        self._run_once()

        return self.work_sequence_range(0,self.work_count)


    def MPI_Work(self, sid0_list, sid1_list):
        return self.do_work(lmm, sid0_list, sid1_list)

    def work_sequence_range(self, start, end):
        self._run_once()

        lmm = self.lmm_from_cache_file()
        lmm.sety(self.pheno['vals'])

        for sid0_list, sid1_list in self.pair_block_sequence_range(start,end):
            yield lambda lmm=lmm,sid0_list=sid0_list,sid1_list=sid1_list : self.do_work(lmm,sid0_list,sid1_list)  # the 'lmm=lmm,...' is need to get around a strangeness in Python



    def reduce(self, result_sequence):
        #doesn't need "run_once()"

        frame = pd.concat(result_sequence)
        frame.sort("PValue", inplace=True)
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
    
    def pair_block_sequence_range(self,block_start,block_end):
        self._run_once()
        assert 0 <= block_start and block_start <= block_end and block_end <= self.work_count, "real assert"

        block_index = block_start
        start = block_index * self.pair_count // self.work_count
        next_start = (block_index+1) * self.pair_count // self.work_count
        size_goal = next_start - start
        end = block_end * self.pair_count // self.work_count

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
        print "Loading precomputation from ", self.cache_file

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

            tf = time.time()
            lmm = LMM()
            t1 = time.time()
            G0_standardized = self.G0.read().standardize()

            print ">G0: ", G0_standardized.val.shape

            t2 = time.time()
            lmm.setG(G0_standardized.val, self.G1val_or_none, a2=self.mixing)
            
            print "XXXXX========XXXXX %lu (Standardized %lu, setG %lu) XXXXX========XXXXX" % (time.time() - tf, t2 - t1, time.time() - t2)


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


    do_pair_count = 0
    do_pair_time = time.time()

    def do_work(self, lmm, sid0_list, sid1_list):
        t1 = time.time()

        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))

        '''
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
        '''

        dict_list = []        

        #This is some of the code for a different way that reads and dot-products 50% more, but does less copying. Seems about the same speed
        #sid0_index_list = self.test_snps.sid_to_index(sid0_list)
        #sid1_index_list = self.test_snps.sid_to_index(sid1_list)
        #sid_index_union_dict = {}
        #sid0_index_index_list = self.create_index_index(sid_index_union_dict, sid0_index_list)
        #sid1_index_index_list = self.create_index_index(sid_index_union_dict, sid1_index_list)
        #snps0_read = self.test_snps[:,sid0_index_list].read().standardize()
        #snps1_read = self.test_snps[:,sid1_index_list].read().standardize()

        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()

        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)

        products = snps_read.val[:,sid0_index_list] * snps_read.val[:,sid1_index_list] # in the products matrix, each column i is the elementwise product of sid i in each list
        X = np.hstack((self.covar, snps_read.val, products))

        k = lmm.S.shape[0]
        N = X.shape[0]

        gpu = False
        
        self.lock.acquire()
        if (k<N):
            gpu_memory_need = 3*X.nbytes + lmm.U.T.nbytes + lmm.U.nbytes      
        else:
            gpu_memory_need = 2*X.nbytes + lmm.U.T.nbytes

        if gpu_memory_need < self.multi_gpu_free[self.my_gpu]:
            self.multi_gpu_free[self.my_gpu] -= gpu_memory_need
            gpu = True
        self.lock.release()
        
        #print "[%d] Need: %d, Free: %d" % (self.myproc, gpu_memory_need, self.gpu_free.value)        
        
        if gpu:
            #print "[%d] GPU RUN" % (self.myproc)
            U_T = np.copy(lmm.U.T, "F")

            t_m = time.time()

            X_gpu   = gpuarray.to_gpu(X)
            U_T_gpu = gpuarray.to_gpu(U_T)
            
            
            UX_gpu  = culinalg.dot(U_T_gpu, X_gpu)
            UX      = UX_gpu.get()

            #print "[%d] Timing Mult X(%d x %d) . U_T(%d x %d), Timing: %0.2f (s)" % (self.id_w, len(X), len(X[0]), len(U_T), len(U_T[0]), time.time() - t_m)

            #print "X:[%d x %d] . U_T:[%d x %d] = UX:%d x %d" % ()
            
            if (k<N):
                #UUX = X - lmm.U.dot(UX)
                print "in"

                U_gpu   = gpuarray.to_gpu(lmm.U)
                UUX_gpu = culinalg.dot(U_gpu, UX_gpu)
                UUX     = X - UUX_gpu.get()
                #print "U:[%d x %d] . UX:[%d x %d] = UUX:%d x %d" % (len(X), len(X[0]), len(U_T), len(U_T[0]), len(UX), len(UX[0]))

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
            self.multi_gpu_free[self.my_gpu] += gpu_memory_need
            self.lock.release()
        else:
            UX = lmm.U.T.dot(X)

            if (k<N):
                UUX = X - lmm.U.dot(UX)
            else:
                UUX = None
        ###--------------------------- CUDA Version -------------------------##
        tt1 = time.time()
        global_vars.first_section += (tt1 - t1)


        t2 = time.time()
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
        global_vars.second_section += (time.time() - t2)

        return dataframe

if __name__ == "__main__":
    
    import doctest
    doctest.testmod()

    print "done"

