FROM centos/python-36-centos7:latest
USER root
RUN ls
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install virtualenv
RUN virtualenv venv
RUN source venv/bin/activate
RUN pip3 install diskcache
RUN pip3 install scikit-learn==0.23.2
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install pytest
RUN pip3 install PyYAML
RUN pip3 install parse
RUN pip3 install Flask
RUN pip3 install jsonpickle
RUN pip3 install requests
RUN pip3 install scipy==1.4.1
RUN pip3 install environs
RUN pip3 install pathlib2
RUN pip3 install diffprivlib
RUN pip3 install psutil
RUN pip3 install setproctitle
RUN pip3 install tabulate
RUN pip3 install lz4
# RUN pip3 install opencv-python
RUN pip3 install gym
RUN pip3 install cloudpickle==1.3.0
RUN pip3 install skorch
RUN pip3 install h5py==2.10.0
RUN pip3 install image
RUN pip3 install aif360
RUN pip3 install fairlearn
RUN pip3 install gensim==3.8.3
RUN pip3 install ray
# RUN pip3 install dm-tree
RUN pip3 install GPUtil
RUN pip3 install paramiko
RUN pip3 install http://github.com/IBM/pycloudmessenger/archive/v0.8.0.tar.gz
RUN pip3 install jupyter
RUN pip3 install kubernetes
RUN pip3 install openshift
RUN pip3 install torch>=1.6.0
RUN pip3 install torchvision>=0.7.0
RUN pip3 install termcolor

RUN yum install -y tmux

WORKDIR /FL
COPY . /FL
CMD ["/bin/bash"]