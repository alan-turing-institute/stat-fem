# DockerFile for stat-firedrake

FROM firedrakeproject/firedrake:latest

# This DockerFile is looked after by
MAINTAINER Eric Daub <edaub@turing.ac.uk>

USER firedrake
WORKDIR /home/firedrake/firedrake/src
RUN bash -c "source /home/firedrake/firedrake/bin/activate; git clone https://github.com/alan-turing-institute/stat-fem.git; cd stat-fem; python setup.py develop"

USER firedrake
WORKDIR /home/firedrake
