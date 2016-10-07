# Fast-LMM Epistasis MPI

MPI implementation for Epistasis Fast-LMM for Linux

CONTACT
-------
  You can contact any of the following developers:

    * Héctor Matínez (martneh@uji.es)

DOWNLOAD and BUILDING
---------------------
  Follow the same steps that Fast-LMM section "Detailed Package Install Instructions" (see "https://github.com/MicrosoftGenomics/FaST-LMM") for install python package
  and then run:

    $ sudo python setup.py install

RUNING
-------
    $ mpirun -np <num_MPI_procs> -host <hosts> python mpi_manager.py <num_python_procs> <epiDataset> <epiPheno>
 	
    $ mpirun -np 4 -hosts host0,host1,host2,host0 python mpi_manager.py 8 epiData epiPheno.pheno

  The results are stored in "dataframe.out" binary file. If you want to read it, you need make a script to it. 
  See the example "output_read.py" to read the file and run it for view the results output.

    $ pyhon output_read.py
