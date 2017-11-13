from fastlmmhpc.util.runner import *
import logging
import fastlmmhpc.pyplink.plink as plink
from pysnptools.snpreader import Pheno
import pysnptools.util as pstutil
import fastlmmhpc.util.util as flutil
import numpy as np
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmmhpc.util.pickle_io import load, save
import time
import pandas as pd
from fastlmmhpc.inference.lmm_cov import LMM as fastLMM
import warnings
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData
from pysnptools.standardizer import Unit
from pysnptools.standardizer import Identity as SS_Identity
from pysnptools.standardizer import DiagKtoN
from pysnptools.standardizer import UnitTrained
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import KernelNpz
from fastlmmhpc.util.mapreduce import map_reduce
from pysnptools.util import create_directory_if_necessary
from pysnptools.snpreader import wrap_matrix_subset
from pysnptools.util.intrangeset import IntRangeSet
from fastlmmhpc.inference.fastlmm_predictor import _snps_fixup, _pheno_fixup, _kernel_fixup, _SnpTrainTest
import fastlmmhpc.inference.linear_regression as lin_reg
from fastlmmhpc.association.single_snp import _set_block_size

def single_snp_linreg(test_snps, pheno, covar=None, max_output_len=None, output_file_name=None, GB_goal=None, runner=None):
    """
    Function performing single SNP GWAS using linear regression. Will reorder and intersect IIDs as needed.

    :param test_snps: SNPs to test. Can be any :class:`.SnpReader`. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs with missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a :class:`.SnpReader` or a string

    :param covar: covariate information, optional: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type covar: a :class:`.SnpReader` or a string


    :param max_output_len: Maximum number of Pvalues to return. Default to None, which means 'Return all'.
    :type max_output_len: number
    
    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks of size iid_count,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"


    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmmhpc.association import single_snp_linreg
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp_linreg(test_snps="../feature_selection/examples/toydata.5chrom", pheno=pheno_fn)
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_576 1e-07 10000


    """
    assert test_snps is not None, "test_snps must be given as input"
    test_snps = _snps_fixup(test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)
    test_snps, pheno, covar  = pstutil.intersect_apply([test_snps, pheno, covar])
    logging.debug("# of iids now {0}".format(test_snps.iid_count))

    _, _, block_size = _set_block_size(test_snps, None, 0, GB_goal, force_full_rank=False, force_low_rank=False)

    #!!!what about missing data in covar, in test_snps, in y
    covar = np.c_[covar.read(view_ok=True,order='A').val,np.ones((test_snps.iid_count, 1))]  #view_ok because np.c_ will allocation new memory
    y =  pheno.read(view_ok=True,order='A').val #view_ok because this code already did a fresh read to look for any missing values

    def mapper(start):
        snp_index = np.arange(start,min(start+block_size,test_snps.sid_count))
        x = test_snps[:,start:start+block_size].read().standardize().val
        _,pval_in = lin_reg.f_regression_cov_alt(x,y,covar)
        pval_in = pval_in.reshape(-1)

        if max_output_len is None:
            return pval_in,snp_index
        else: #We only need to return the top max_output_len results
            sort_index = np.argsort(pval_in)[:max_output_len]
            return pval_in[sort_index],snp_index[sort_index]

    def reducer(pval_and_snp_index_sequence):
        pval_list = []
        snp_index_list = []
        for pval, snp_index in pval_and_snp_index_sequence:
            pval_list.append(pval)
            snp_index_list.append(snp_index)
        pval = np.concatenate(pval_list)
        snp_index = np.concatenate(snp_index_list)
        sort_index = np.argsort(pval)
        if max_output_len is not None:
            sort_index = sort_index[:max_output_len]
        index = snp_index[sort_index]

        dataframe = pd.DataFrame(
            index=np.arange(len(index)),
            columns=('sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue')
            )
        #!!Is this the only way to set types in a dataframe?
        dataframe['sid_index'] = dataframe['sid_index'].astype(np.float)
        dataframe['Chr'] = dataframe['Chr'].astype(np.float)
        dataframe['GenDist'] = dataframe['GenDist'].astype(np.float)
        dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
        dataframe['PValue'] = dataframe['PValue'].astype(np.float)

        dataframe['sid_index'] = index
        dataframe['SNP'] = test_snps.sid[index]
        dataframe['Chr'] = test_snps.pos[index,0]
        dataframe['GenDist'] = test_snps.pos[index,1]
        dataframe['ChrPos'] = test_snps.pos[index,2]
        dataframe['PValue'] = pval[sort_index]

        if output_file_name is not None:
            dataframe.to_csv(output_file_name, sep="\t", index=False)

        return dataframe

    dataframe = map_reduce(xrange(0,test_snps.sid_count,block_size),
                           mapper=mapper,
                           reducer=reducer,
                           input_files=[test_snps,pheno,covar],
                           output_files=[output_file_name],
                           name = "single_snp_linreg",
                           runner=runner)
    return dataframe

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()

