# Improvements For Fast-LMM Epistasis (Multi-GPU)

Improvements for Fast-LMM Epistasis (Multi-GPU version)

CONTACT
-------
  You can contact any of the following developers:

    * Héctor Matínez (martneh@uji.es)

DOWNLOAD and BUILDING
---------------------
  Clone the repository.
  
    $ git clone https://github.com/epiproject/FaST-LMM-HPC.git

  Select your branch (in this case 'multi-gpu').  

    $ git checkout multi-gpu

  Follow the same steps that Fast-LMM section "Detailed Package Install Instructions" (see "https://github.com/MicrosoftGenomics/FaST-LMM") for install python package.
  Also you need install the package:

    pycuda
    skcuda

  Afther that, run:

    $ sudo python setup.py install

RUNING
-------
  Follow the same steps that Fast-LMM url or run:

    $ python run_epi.py <num_python_procs> <epiDataset> <epiPheno>
    
    $ python run_epi.py 8 epiData epiPheno.pheno

