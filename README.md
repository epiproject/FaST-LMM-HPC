# Example to run Fast-LMM Epistasis MPI

$ mpirun -np 4 -hosts host0,host1,host2,host0 python mpi_manager.py 8 epiData epiPheno.pheno

The results are stored in "dataframe.out" binary file. If you want to read it, you need make a script to read it.
Run output_read.py

$ pyhon output_read.py
