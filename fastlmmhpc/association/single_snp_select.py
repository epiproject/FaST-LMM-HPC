import numpy as np
import logging
from fastlmmhpc.association import single_snp
from sklearn.cross_validation import KFold
import pandas as pd
import os
import time

import pysnptools.util as pstutil
from fastlmmhpc.inference import FastLMM
from fastlmmhpc.util.mapreduce import map_reduce
from fastlmmhpc.inference.fastlmm_predictor import _snps_fixup, _pheno_fixup, _kernel_fixup
from fastlmmhpc.association.single_snp import _K_per_chrom
from pysnptools.standardizer import Unit

from pysnptools.kernelreader import KernelReader
from pysnptools.kernelreader import KernelData
from fastlmmhpc.association import single_snp_linreg

from fastlmmhpc.util.mapreduce import map_reduce
from fastlmmhpc.association.single_snp_all_plus_select import _kfold
from fastlmmhpc.util.runner import Local
from pysnptools.kernelreader import Identity as KernelIdentity


def _fixup(test_snps, G, pheno, covar):
    test_snps = _snps_fixup(test_snps)
    G = _snps_fixup(G or test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)
    G, test_snps, pheno, covar  = pstutil.intersect_apply([G, test_snps, pheno, covar])
    return test_snps, G, pheno, covar

def single_snp_select(test_snps, pheno, G=None, covar=None,
                 k_list = None,
                 n_folds=10, #1 is special and means test on train
                 just_return_selected_snps=False,
                 seed = 0, output_file_name = None,
                 GB_goal=None, force_full_rank=False, force_low_rank=False, h2=None, runner=None):
    """
    Function performing single SNP GWAS based on covariates (often PCs) and a similarity matrix constructed of the top *k* SNPs where
    SNPs are ordered via the PValue from :meth:`.single_snp_linreg` and *k* is determined via out-of-sample prediction. Will reorder and intersect IIDs as needed.

    :param test_snps: SNPs to test. Can be any :class:`.SnpReader`. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs with missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a :class:`.SnpReader` or a string

    :param G: SNPs from which to create a similarity matrix of the top *k* SNPs. If not given, will use test_snps.
           Can be any :class:`.SnpReader`. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G: :class:`.SnpReader` or a string

    :param covar: covariate information, optional: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type covar: a :class:`.SnpReader` or a string

    :param k_list: Values of *k* (in addition to 0) to test. Default to [1,2,4,8,...8192].
    :type k_list: list of numbers

    :param n_folds: Number of folds of cross validation to use for out-of-sample evaluation of various values of *k*. Default to 10.
    :type n_folds: number
    
    :param just_return_selected_snps: Instead of returning the results of GWAS, return the top *k* SNPs selected.
    :type just_return_selected_snps: list of strings

    :param seed: (optional) Random seed used to generate permutations for lrt G0 fitting.
    :type seed: number

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param force_full_rank: Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
    :type force_full_rank: Boolean

    :param force_low_rank: Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
    :type force_low_rank: Boolean

    :param h2: A parameter to LMM learning that tells how much weight to give the K's vs. the identity matrix, optional
            If not given will search for best value.
    :type h2: number

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"


    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmmhpc.association import single_snp_select
    >>> from pysnptools.snpreader import Bed
    >>> from fastlmmhpc.util import compute_auto_pcs
    >>> logging.basicConfig(level=logging.INFO)
    >>> bed_fn = "../../tests/datasets/synth/all.bed"
    >>> phen_fn = "../../tests/datasets/synth/pheno_10_causals.txt"
    >>> covar = compute_auto_pcs(bed_fn)
    >>> results_dataframe = single_snp_select(test_snps=bed_fn, G=bed_fn, pheno=phen_fn, covar=covar, GB_goal=2)
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    snp495_m0_.01m1_.04 0.0 5000

    """

    #!!!code similar to single_snp and feature_selection
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    assert test_snps is not None, "test_snps must be given as input"

    if k_list is None:
        k_list = np.logspace(start=0, stop=13, num=14, base=2)

    test_snps, G, pheno, covar = _fixup(test_snps, G, pheno, covar)
    common_input_files = [test_snps, G, pheno, covar]

    k_list_in = [0] + [int(k) for k in k_list if 0 < k and k <= G.sid_count]

    def top_snps_for_each_fold_nested(kfold_item):
        fold_index, (train_idx, test_idx) = kfold_item
        _, G_in, pheno_in, covar_in = _fixup(test_snps, G, pheno, covar)
        nested = single_snp_linreg(G_in[train_idx,:],pheno_in[train_idx,:],covar_in[train_idx,:],GB_goal=GB_goal,max_output_len=max(k_list_in))
        return nested
    def top_snps_for_each_fold_reducer(dataframe_list):
        result = [list(dataframe['SNP']) for dataframe in dataframe_list]
        return result


    #Find top snps for each fold        
    fold_index_to_top_snps = map_reduce(_kfold(G.iid_count, n_folds, seed, end_with_all=True,iid_to_index=G.iid_to_index),
                                         nested=top_snps_for_each_fold_nested,
                                         reducer=top_snps_for_each_fold_reducer,
                                         name = "top_snps_for_each_fold",
                                         input_files = common_input_files,
                                         runner=runner)

    #=================================================
    # Start of definition of inner functions
    #=================================================
    def k_index_to_nLL_mapper(k):
        _, G_in, pheno_in, covar_in = _fixup(test_snps, G, pheno, covar)
        nll_sum=0
        mse_sum = 0
        n_folds_in = 0
        for fold_index, (train_idx, test_idx) in _kfold(G.iid_count, n_folds, seed, end_with_all=False,iid_to_index=G.iid_to_index):
            assert set(train_idx).isdisjoint(set(test_idx)), "real assert"
            top_snps_in_fold = fold_index_to_top_snps[fold_index][:k]
            sid_idx_in_fold = G_in.sid_to_index(top_snps_in_fold)
            G_train = G_in[train_idx,sid_idx_in_fold] if k > 0 else None
            fastlmm = FastLMM(force_full_rank=force_full_rank, force_low_rank=force_low_rank,GB_goal=GB_goal)
            fastlmm.fit(K0_train=G_train, X=covar_in[train_idx,:], y=pheno_in[train_idx,:], h2=h2) #iid intersection means when can give the whole covariate and pheno
            G_test = G_in[test_idx,sid_idx_in_fold] if k > 0 else KernelIdentity(G_in.iid,G_in.iid[test_idx]) #!!! instead of this, which blows up when # of iids is large, should switch to linear regression model with k is 0
            nll,mse = fastlmm.score(K0_whole_test=G_test,X=covar_in[test_idx,:],y=pheno_in[test_idx,:],return_mse_too=True) #iid intersection means when can give the whole covariate and pheno
            nll_sum += nll
            mse_sum += mse
            n_folds_in += 1
        logging.info("k={0},nLL={1},average mse={2}".format(k,nll_sum,mse_sum / n_folds_in))
        return nll_sum
    #=================================================
    # End of definition of inner functions
    #=================================================


    #find best # of top SNPs
    k_index_to_nLL = map_reduce(k_list_in,
                                mapper = k_index_to_nLL_mapper,
                                input_files = common_input_files,
                                name = "k_index_to_nLL",
                                runner=runner)
    best_k = k_list_in[np.argmin(k_index_to_nLL)]

    top_snps = fold_index_to_top_snps[-1][:best_k]
    if just_return_selected_snps:
        return top_snps

    sid_idx = G.sid_to_index(top_snps)
    G_top = G[:,sid_idx]

    # Run GWAS with leave-one-chrom out
    single_snp_result = single_snp(test_snps=test_snps, K0=G_top, pheno=pheno,
                                covar=covar, leave_out_one_chrom=True,
                                GB_goal=GB_goal,  force_full_rank=force_full_rank, force_low_rank=force_low_rank, h2=h2,
                                output_file_name=output_file_name,runner=runner)

    return single_snp_result

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()

