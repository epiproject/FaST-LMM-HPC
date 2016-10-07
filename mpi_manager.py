from mpi4py import MPI
import sys
import multiprocessing
import time
from fastlmm.association import epistasis    
import pandas as pd
import numpy as np
import pickle

######################### MERGE RESULTS FIRST VERSION ###########################

def create_sub_result_sequence(result_sequence):
    for result in result_sequence:
        #workIndex = result.pop(0)
        for sub_workIndex, pair in result:
            yield pair

#################################################################################

def merge_sequence_from_disk(mpi_procs, taskcount):
    for r in range(mpi_procs):
        for taskindex in range(taskcount):
            task_file_name = ".working."+str(r)+"/"+str(taskindex)+"."+str(taskcount)+".p"
            print task_file_name
            with open(task_file_name, mode='rb') as f:
                result = pickle.load(f)
            yield result

#################################################################################
    
def factorial(n):
    if n == 0:
        return 1
    else:
        res = 1
        for n in range(1, n + 1):
            res = res * n
        return res

#################################################################################

pairs_per_block = 1000
comm            = MPI.COMM_WORLD
rank            = comm.Get_rank()
mpi_procs       = comm.Get_size()
taskcount       = int(sys.argv[1])
snps_fd         = sys.argv[2]
pheno_fd        = sys.argv[3]

#print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

if rank == 0:
    final_t = time.time()

if rank == mpi_procs - 1:
    snps_list = []
    fd = open(snps_fd + ".bim", "r")
    for l in fd:
        spl = l.split("\t")
        snps_list.append(spl[1])

    snps = len(snps_list)
    
    ####################################################################################

    total_pairs    = factorial(snps) / (2 * factorial(snps - 2))
    row   = 0
    start = 1

    pairs = 0
    task_queue = []

    task_queue.append([0, 0])
    for r in range (0, snps - 1):
        for c in range (r + 1, snps):
            pairs += 1
            if pairs == pairs_per_block:
                if (r + 1) != (snps - 1):
                    task_queue.append([r, c])
                pairs = 0
    
    ####################################################################################

    actual_task    = 0
    n_task         = len(task_queue)
    rank_finished  = 0

    while True:
        rank_dest = comm.recv(source=MPI.ANY_SOURCE, tag=1)

        if rank_dest == -1:
            rank_finished += 1
            if rank_finished == mpi_procs - 1:
                break
            else:
                continue

        if actual_task < n_task:
            task = task_queue[actual_task]
            actual_task += 1
        else:
            task = []
            
        comm.send(task, dest=rank_dest, tag=1)

    ####################################################################################

else:
    print "[%d]WORKER PROCESS START" % (rank)
    t = time.time()
    new_res = epistasis(snps_fd, pheno_fd, snps_fd, taskcount, pairs_per_block=pairs_per_block)
    print "# WORKER [%d] FINISHED IN %0.2f (S) #" % (rank, time.time() - t)

comm.Barrier()

if rank == 0:
    result_sequence = merge_sequence_from_disk(mpi_procs - 1, taskcount)
    result = create_sub_result_sequence(result_sequence)

    frame = pd.concat(result)
    frame.sort("PValue", inplace=True)
    frame.index = np.arange(len(frame))

    file_name = "dataframe.out"
    f = open(file_name, "wb")

    pickle.dump(frame, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    #print frame
    #print "#### FINISHED IN %0.2f (S) ####" % (time.time() - final_t)
    #for index, row in frame.iterrows():
    #    print row['SNP0'],row['SNP1'],row['PValue']

    #print frame

    #pd.set_option('display.max_rows', len(frame))
    #print(frame)
    #pd.reset_option('display.max_rows')
     


MPI.Finalize()
