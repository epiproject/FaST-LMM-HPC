# Improvements For Fast-LMM Epistasis

Improvements for Fast-LMM Epistasis

CONTACT
-------
  You can contact any of the following developers:

    * Héctor Matínez (martneh@uji.es)

DOWNLOAD and BUILDING
---------------------
  Clone the repository.
  
    $ git clone https://github.com/epiproject/FaST-LMM-HPC.git

  Select your branch (in this case 'master').  

    $ git checkout master 

  Follow the same steps that Fast-LMM section "Detailed Package Install Instructions" (see "https://github.com/MicrosoftGenomics/FaST-LMM") for install python packages.
  Also you need install the packages:

    pycuda
    skcuda

  Afther that, run:

    $ sudo python setup.py install

RUNING
-------
  Follow the same steps that Fast-LMM url or run:

    $ python run_epi.py <num_python_procs> <epiDataset> <epiPheno>
    
    $ python run_epi.py 8 epiData epiPheno.pheno

  You can find some epistasis datasets in "./datastes/" path. For example, if you want to execute FaST-LMM 'all_chr.maf0.001.N300' dataset and 12 processes you can run:

    $ python run_epi.py 12 ./datastes/all_chr.maf0.001.N300 ./datastes/phenSynthFrom22.23.N300.txt
