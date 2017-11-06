# Fast-LMM Epistasis MPI

MPI implementation for Epistasis Fast-LMM for Linux

CONTACT
-------
  You can contact any of the following developers:

    * Héctor Matínez (martneh@uji.es)

DOWNLOAD and BUILDING
---------------------
  Clone the repository.
  
    $ git clone https://github.com/epiproject/FaST-LMM-HPC.git

  Select your branch (in this case 'mpi-multi-gpu').  

    $ git checkout mpi-multi-gpu

  Follow the same steps that Fast-LMM section "Detailed Package Install Instructions" (see "https://github.com/MicrosoftGenomics/FaST-LMM") for install python package.
  Also you need install the package:

    pycuda
    skcuda
    mpi4py    

  Afther that, run:

    $ sudo python setup.py install

RUNING
-------
  You can run the next command line:
 
    $ mpirun -np <num_MPI_procs> -host <hosts> python mpi_manager.py <num_python_procs> <epiDataset> <epiPheno>
 	
  Example to run the MPI FaST-LMM version with 3 workers (you must set an aditional process for the manager, in this case 4 process): 

    $ mpirun -np 4 -hosts host0,host1,host2,host0 python mpi_manager.py 8 epiData epiPheno.pheno

  The results are stored in "dataframe.out" binary file. If you want to read it, you need make a script for that.
  You can found in the path an example of script "output_read.py" for read the output file. This script store the dataframe generated by the application
  in the 'result' variable, and then show the data in the screen.

    $ pyhon output_read.py
