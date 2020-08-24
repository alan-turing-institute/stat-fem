# DockerFile for stat-firedrake

FROM firedrakeproject/firedrake:latest

# This DockerFile is looked after by
MAINTAINER Eric Daub <edaub@turing.ac.uk>

USER firedrake
WORKDIR /home/firedrake/firedrake/src
RUN bash -c "source /home/firedrake/firedrake/bin/activate && \
             git clone https://github.com/alan-turing-institute/stat-fem.git && \
             cd stat-fem && \
             git checkout basicfunc && \
             pip install -r requirements.txt && \
             pip install -r requirements-dev.txt && \
             python setup.py develop"

RUN echo "source /home/firedrake/firedrake/bin/activate" >> /home/firedrake/.bashrc

USER firedrake
WORKDIR /home/firedrake/firedrake/src/stat-fem/stat_fem/tests
ENTRYPOINT /bin/bash