import numpy as np
import scipy as sp
import sys
import logging
import time
from fastlmmhpc.external.pca import PCA
from pysnptools.snpreader import Bed


def compute_auto_pcs(snpreader, cutoff=.1, k_values=np.arange(11), output_file_name=None):
    """
    Function automatically finds the best principle components (PCs)

    :param snpreader: SNPs for which to find the best PCs
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type snpreader: a :class:`.SnpReader` or a string

    :param cutoff: (default: .1) The degree of relatedness to remove before finding the best number of PCs.
        Relatedness is measured with a RRM (realized relationship matrix) so 0 is no relation, .5 is a sibling or parent, and 1 is self or twin.
    :type cutoff: a number between 0 and 1.

    :param k_values: (default: 0 ... 10 [inclusive]) The number of PCs to search.
    :type k_values: list of integers

    :rtype: A phenotype dictionary with property 'iid' listing the iids and property 'vals' containing a nparray of PC values.

    :Example:

    >>> import logging
    >>> from fastlmmhpc.util import compute_auto_pcs
    >>> logging.basicConfig(level=logging.INFO)
    >>> file_name = "../feature_selection/examples/toydata"
    >>> best_pcs = compute_auto_pcs(file_name)
    >>> print int(best_pcs['vals'].shape[0]),int(best_pcs['vals'].shape[1])
    500 0

    """
    #!!could use regression tests beyond the docttest

    snpreader = _snp_fixup(snpreader)

    logging.info("reading all_std_snpdata")
    all_std_snpdata = snpreader.read().standardize() #!!could doing C or F be better?

    #use vertex cut to find just parents
    logging.info("Finding relatedness of all iids")
    from pysnptools.standardizer import Identity
    rrm = all_std_snpdata.kernel(Identity()) / snpreader.sid_count
    import fastlmmhpc.util.VertexCut as vc
    remove_set = set(vc.VertexCut().work(rrm,cutoff)) #These are the indexes of the IIDs to remove
    logging.info("removing {0} of {1} iids".format(len(remove_set), snpreader.iid_count))
    keep_list = [x for x in xrange(all_std_snpdata.iid_count) if x not in remove_set]
    nofam_snpreader = all_std_snpdata[keep_list,:]
    #nofam_snpreader = all_std_snpdata#[1:,:]
    #print "#!!warning skipping  vertext cut"
    
    #learn # of pcs and generate on nonchild view of data
    if max(k_values) >= nofam_snpreader.iid_count or max(k_values) >= nofam_snpreader.sid_count: 
        raise Exception("The number of PCs search should be less than the # of rows and also the # of cols in the matrix after near relatives are removed")
    randomstate = 1

    from sklearn.cross_validation import KFold
    n_folds = 10
    folds = KFold(nofam_snpreader.sid_count, n_folds = n_folds, shuffle=True, random_state=randomstate)
        
    scores = np.zeros((k_values.shape[0],n_folds))
    for i_fold, [train_idx,test_idx] in enumerate(folds):
        Utr,Str,Vtr,mean = None, None, None, None
        logging.info('test set size: {0}'.format(len(test_idx)))

        logging.info('creating X_train for fold {0}'.format(i_fold))
        t0 = time.time()
        X_train = nofam_snpreader[:,train_idx].read(order='F').val.T#!!would order='C' be faster?
        logging.info("done after %.4f seconds" % (time.time() - t0))

        logging.info('creating X_test for fold {0}'.format(i_fold))
        t0 = time.time()
        X_test = nofam_snpreader[:,test_idx].read(order='F').val.T
        logging.info("done after %.4f seconds" % (time.time() - t0))

        logging.info('Creating svd')
        t0 = time.time()
        for i_k, k in enumerate(k_values):
            pca = PCA(n_components = k, copy=False)
            Utr,Str,Vtr,mean = pca._fit(X_train,Utr,Str,Vtr,mean)
            if t0 is not None:
                logging.info("done after %.4f seconds" % (time.time() - t0))
                t0 = None
            scores[i_k,i_fold] = pca.score(X_test)
            logging.info("{0},{1},{2}".format(i_fold, k, scores[i_k,i_fold]))

    normalizedMean = scores.mean(axis=1)

    logging.info('normalized Means: {0}'.format(normalizedMean))
    
    bestNumPCs = k_values[normalizedMean.argmax()]
    logging.info('best num PCs: {0}'.format(bestNumPCs))

    
    logging.info("computing svd...")
    t0 = time.time()
    Utr,Str,Vtr,mean = None, None, None, None
    pca = PCA(n_components = bestNumPCs,copy=False)
    nofam_snpdata = nofam_snpreader.read(order='F') #!!would order='C' be faster?
    pca._fit(nofam_snpdata.val,Utr,Str,Vtr,mean) #This will zero-center nofam_snpdata.val, so don't use after this point.
    t1 = time.time()
    logging.info("done after %.4f seconds" % (t1 - t0))
    
    #apply those pcs to all the data (i.e.  transform)
    logging.info('Projecting individuals to PCs space...')
    X_fit = pca.transform(all_std_snpdata.val)
    
    ##write the file out
    if output_file_name is not None:
        logging.info('writing results to file...')
        with open(output_file_name, 'w') as f:
            for iid_index, (famid,indid) in enumerate(all_std_snpdata.iid):
                f.write("{0} {1} ".format(famid,indid))
                f.write(' '.join([str(pc) for pc in X_fit[iid_index, :]]))
                f.write('\n')

    result = {'iid':sp.array(snpreader.iid),'vals':X_fit, 'header':["pc_{0}".format(index) for index in xrange(bestNumPCs)]}
    return result

def _snp_fixup(snp_input):
    if isinstance(snp_input, str):
        return Bed(snp_input)
    else:
        return snp_input


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import logging
    from pysnptools.snpreader import Bed
    from fastlmmhpc.util import compute_auto_pcs
    logging.basicConfig(level=logging.INFO)
    #file_name = "../../tests/datasets/mouse/alldata"
    file_name = "../feature_selection/examples/toydata"
    #file_name = r"c:\deldir\N4000S50000c500h0.50s0.00p0.50F0.0050FH0.2000v0.30_3"
    result = compute_auto_pcs(file_name)
    #print result


    print "done"

