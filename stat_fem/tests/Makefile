.PHONY: tests

tests:
	pytest --cov=..
	mpiexec -n 2 python -m pytest --with-mpi
	mpiexec -n 4 python -m pytest --with-mpi
