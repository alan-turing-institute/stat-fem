.. _installation:

Installation
============

Installing stat-fem
-------------------

Installing Firedrake
~~~~~~~~~~~~~~~~~~~~

``stat-fem`` requires a working Firedrake installation. Firedrake requires
compliation of a number of complex dependencies, so is most easily done
by following instructions on the
`Firedrake homepage <https://www.firedrakeproject.org>`_. The provided
install script works on Ubuntu Linux systems as well as MacOS. For other
systems, you can use the provided Docker_ image.

To automatically install ``stat-fem`` inside of the Firedrake install, you
can call the Firedrake installation script with the following options: ::

   curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
   python3 firedrake-install --install git+https://github.com/alan-turing-institute/stat-fem#egg=stat-fem

Otherwise, follow the instructions below to install ``stat-fem``.

**NOTE:** You will need to activate the Firedrake virtual environment
prior to executing any of the following commands. From the directory
where you ran the Firedrake installation script, enter: ::

   . firedrake/bin/activate

into your terminal (note that if you are using ``csh``, the activation
script is different). This should correctly set your path and ensure
that further installation commands install into the correct environment.

Requirements
~~~~~~~~~~~~

Firedrake and ``stat-fem`` require Python 3.6 or later. Working Numpy and
Scipy installations are required, though these are automatically installed
inside the Firedrake virtual environment. You should be able to install
these packages using ``pip`` if they are not already available. From the
base ``stat-fem`` directory, you can install all required packages using: ::

   pip install -r requirements.txt

Installation
~~~~~~~~~~~~

Then to install the main code, run the following command from the base
``stat-fem`` directory: ::

   python setup.py install

This will install the main code in the system Python installation.
You may need adminstrative priveleges to install the software itself,
depending on your system configuration. However, any updates to the code
cloned through the github repository (particularly if you are using the
devel branch, which is under more active development) will not be reflected
in the installation using this method. If you would like
to always have the most active development version, install using: ::

   python setup.py develop

This will insert symlinks to the repository files into the system Python installation so that files
are updated whenever there are changes to the code files.

Documentation
-------------

The code documentation is available on
`readthedocs <https://stat-fem.readthedocs.io>`_. A current
build of the ``master`` and ``devel`` branches should always be available
in HTML or PDF format.

To build the documentation yourself requires Sphinx, which can be installed
using ``pip``. This can also be done in the ``docs`` directory using
``pip install -r requirements.txt``. To build the documentatation,
change to the ``docs`` directory. There is a Makefile in the `docs` directory
to facilitate building the documentation for you. To build the HTML version,
enter the following into the shell from the ``docs`` directory: ::

   make html

This will build the HTML version of the documentation. A standalone PDF
version can be built, which requires a standard LaTeX installation, via: ::

   make latexpdf

In both cases, the documentation files can be found in the corresponding
directories in the ``docs/_build`` directory. Note that if these
directories are not found on your system, you may need to create them in
order for the build to finish correctly. A version of the documentation can
also be found at the link above on Read the Docs.

Testing the Installation
------------------------

``stat-fem`` includes a full set of unit tests. To run the test suite, you
will need to install the development dependencies, which include ``pytest``
and a number of plugins to enable parallel testing and to give coverage
reports, which can be done in the main ``stat-fem`` directory via
``pip install -r requirements-dev.txt``. The ``pytest-cov`` package is not
required to run the test suite, but is useful if you are developing
the code to determine test coverage.

The tests can be run from the base ``stat-fem`` directory or the
``stat_fem/tests`` directory. Simply enter ``pytest`` into the shell,
which will run all tests on a single processor and print out the results
to the console. Note that this will skip a number of tests, which require
running on multiple cores to test the parallel capabilities of ``stat-fem``.
Running the tests in parallel can be done with 2 or 4 cores with the
following command: ::

   mpiexec -n 2 python -m pytest --with-mpi

This will run the test suite on 2 cores, and similarly for 4 cores replace
the "2" with "4" in the above shell command.

In the ``stat_fem/tests`` directory, there is also a ``Makefile`` that will
run the tests for you on 1, 2, and 4 cores in succession.
You can simply enter ``make tests`` into the shell to run the full
serial and parallel suite.

Docker
------

If you are on a system that is not supported by the Firedrake installation
script, you can still use ``stat-fem`` within a Docker container. You
will first need to `install Docker <https://docs.docker.com/engine/install/>`_
and then launch the Docker Engine.

Once Docker is running, navigate to the `Docker` directory of the ``stat-fem``
installation and enter: ::

   docker build -t stat-fem - < Dockerfile

This will pull the Firedrake Docker container and install ``stat-fem`` for you.
Once you have build the image, you can start a shell by entering: ::

   docker run -it stat-fem

The image will have already activated the Firedrake virtual environment
and be ready to run any scripts. Note that any files that you produce
from within the container will not be accessible to the outside world
unless you mount a shared directory. The ``stat-fem`` docker image has
a directory ``/home/firedrake/share``, which you can mount to a shared
directory on your filesystem by running the container as follows: ::

   mkdir share
   docker run -it stat-fem -v share:/home/firedrake/share

Any files copied into ``/home/firedrake/share`` will be available in the
``share`` directory on the host system.
