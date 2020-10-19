# DockerFile for stat-fem

FROM firedrakeproject/firedrake:latest

# This DockerFile is looked after by
MAINTAINER Eric Daub <edaub@turing.ac.uk>

# install stat-fem

USER firedrake
WORKDIR /home/firedrake/firedrake/src
RUN bash -c "source /home/firedrake/firedrake/bin/activate && \
             git clone https://github.com/alan-turing-institute/stat-fem.git && \
             cd stat-fem && \
             pip install -r requirements.txt && \
             pip install -r requirements-dev.txt && \
             python setup.py install"

# create share directory for outputs

USER firedrake
WORKDIR /home/firedrake
RUN bash -c "mkdir -p share"

# set up bashrc to automatically activate virtualenv

RUN echo "source /home/firedrake/firedrake/bin/activate" >> /home/firedrake/.bashrc

USER firedrake
WORKDIR /home/firedrake/
ENTRYPOINT /bin/bash