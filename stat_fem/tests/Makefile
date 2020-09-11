.PHONY: tests

tests:
	pytest --cov=..
	mpiexec -n 2 python -m pytest --with-mpi --cov=.. --cov-append
	mpiexec -n 4 python -m pytest --with-mpi --cov=.. --cov-append
