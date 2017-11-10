# FaST-LMM epistasis test HPC extension (mpi branch)

FaST-LMM-HPC is a High Performance Computing extension of the epistasis test
module of [FaST-LMM](https://github.com/MicrosoftGenomics/FaST-LMM). The next branches (versions) are available:
* __Master__: FaST-LMM with the enhancements described in Martínez et al. (2017).
* __Multi-GPU__: An extension that can make use of multiples GPUs installed on the same node.
* __MPI__: An extension that can be executed on a cluster (using a single GPU per node).
* __MPI-Multi-GPU__: An extension that can be executed on a cluster (with multiple GPUs per node).


Install instructions
---------------------

1) Clone this repository:

    $ git clone https://github.com/epiproject/FaST-LMM-HPC.git

2) Select the desired branch (i.e., 'mpi').

    $ git checkout mpi

3) Install the FaST-LMM dependencies described on the ["Detailed Package Install
Instructions"](https://github.com/MicrosoftGenomics/FaST-LMM) section of
Fast-LMM.

4) In addition to those, install the next two packages that are required by
FaST-LMM-HPC:

* pycuda
* skcuda
* mpi4py

5) Finally, on the directory where you copied the source code of FaST-LMM-HPC,
run:

    $ sudo python setup.py install


Running
-------
  You can use the next command line to execute an standalone epistasis test with `P` MPI processes and `N` Python processes on an `EPISTASIS_DATASET` and an
`EPISTAIS_PHENOTYPE` in the following way:
 
    $ mpirun -np P -host HOSTS python mpi_manager.py N  EPISTASIS_DATASET  EPISTASIS_PHENOTYPE
 	
  For example, to execute an epistasis test with 2 MPI processes (you must set an aditional process for the manager, in this case 3 MPI process), 6 processes on the `all_chr.maf0.001.N300` dataset, available at the `/datasets` subdirectory, the next command should be executed: 

    $ mpirun -np 3 -hosts host0,host1,host0 python mpi_manager.py 6 ./datasets/all_chr.maf0.001.N300 ./datasets/phenSynthFrom22.23.N300.txt

  The results are stored in "dataframe.out" binary file. If you want to read it, you need make a script for that.
  You can found in the path an example of script "output_read.py" for read the output file. This script store the dataframe generated by the application
  in the 'result' variable, and then show the data in the screen.

    $ pyhon output_read.py


Contact
-------

If you have any doubt, please do not hesitate to contact with:
* Héctor Martínez (<martineh@uji.es>)


References
----------

Martínez, H., Barrachina, S., Castillo, M., Quintana-Ortí, E. S., De Argila, J. R., Farré, X., and Navarro, A. 2017. Accelerating FaST-LMM for epistasis tests. In Algorithms and Architectures for Parallel Processing: 17th International Conference, ICA3PP 2017, Helsinki, Finland, August 21-23, 2017, Proceedings, pages 548–557. Springer International Publishing.
