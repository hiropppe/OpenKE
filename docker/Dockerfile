FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN mkdir /root/_INSTALL

WORKDIR /root/_INSTALL

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

ENV TZ Asia/Tokyo
RUN apt-get update \
  && apt-get install -y tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        wget \
        vim \
        git \
        net-tools \
        iputils-ping \
        netcat \
        tcpdump \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        python-wheel \
        python-mock \
        python-pip \
        python-setuptools \
        rsync \
        zip \
        unzip \
        g++ \
        gfortran \
        mlocate \
        libcupti-dev \
        language-pack-ja-base \
        language-pack-ja \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        updatedb

RUN pip install --upgrade pip

RUN pip install Cython

RUN git clone git://github.com/xianyi/OpenBLAS \
 && cd OpenBLAS \
 && make FC=gfortran \
 && make install

RUN git clone git://github.com/numpy/numpy \
 && cd numpy \
 && git checkout v1.14.2 \
 && cp site.cfg.example site.cfg \
 && sed -i 's|# \[openblas\]|\[openblas\]|' site.cfg \
 && sed -i 's|# libraries = openblas|libraries = openblas|' site.cfg \
 && sed -i 's|# include_dirs = /opt/OpenBLAS/include|include_dirs = /opt/OpenBLAS/include|' site.cfg \
 && sed -i 's|# library_dirs = /opt/OpenBLAS/lib|library_dirs = /opt/OpenBLAS/lib|' site.cfg \
 && sed -i 's|# runtime_library_dirs = /opt/OpenBLAS/lib|runtime_library_dirs = /opt/OpenBLAS/lib|' site.cfg \
 && python setup.py config \
 && python setup.py build \
 && python setup.py install

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        h5py==2.7.1 \
        smart-open==1.5.7 \
        tqdm \
        jedi \
        flake8 \
        autopep8 \
        && \
    python -m ipykernel.kernelspec

RUN git clone https://github.com/taku910/mecab && \
    cd mecab/mecab && \
    ./configure --enable-utf-8-only && \
    make && \
    make check && \
    make install && \
    cd python && \
    python setup.py build && \
    python setup.py install

RUN ldconfig

RUN cd mecab/mecab-ipadic && \
    ./configure --with-charset=utf8 && \
    make && \
    make install

# Install Java8
#RUN add-apt-repository ppa:webupd8team/java && \
#    echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
#    apt-get update && apt-get install -y oracle-java8-installer

# Install Bazel for building tensorflow
#RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh && \
#    chmod +x bazel-0.5.4-installer-linux-x86_64.sh && \
#    ./bazel-0.5.4-installer-linux-x86_64.sh --user

# Copy cuDNN
COPY cudnn-9.2-linux-x64-v7.1.solitairetheme8 /root/_INSTALL/

RUN tar xzf cudnn-9.2-linux-x64-v7.1.solitairetheme8 && \
    cp -a cuda/include/* /usr/local/cuda/include/ && \
    cp -a cuda/lib64/* /usr/local/cuda/lib64/

# Install Tensorflow from source
#RUN git clone https://github.com/tensorflow/tensorflow && \
#    cd tensorflow && \
#    git checkout v1.5.0

#RUN cd tensorflow && \
#    export PATH=$PATH:/root/bin && \
#    export PYTHON_BIN_PATH=$(which python) && \
#    export PYTHON_LIB_PATH="/usr/lib/python2.7/dist-packages" && \
#    export CC_OPT_FLAGS="-march=native" && \
#    export TF_NEED_S3="0" && \
#    export TF_NEED_GCP="0" && \
#    export TF_NEED_HDFS="0" && \
#    export TF_NEED_JEMALLOC="1" && \
#    export TF_NEED_OPENCL="0" && \
#    export TF_NEED_OPENCL_SYCL="0" && \
#    export TF_NEED_CUDA="1" && \
#    export TF_ENABLE_XLA="1" && \
#    export TF_NEED_GDR="0" && \
#    export TF_NEED_VERBS="1" && \
#    export TF_CUDA_CLANG="0" && \
#    export TF_CUDA_VERSION="9.1" && \
#    export CUDA_TOOLKIT_PATH="/usr/local/cuda" && \
#    export TF_CUDNN_VERSION="7" && \
#    export CUDNN_INSTALL_PATH="/usr/local/cuda" && \
#    export TF_CUDA_COMPUTE_CAPABILITIES="6.1" && \
#    export GCC_HOST_COMPILER_PATH=$(which gcc) && \
#    export TF_NEED_MPI="0" && \
#    ./configure && \
#    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-msse4.1 --copt=-msse4.2 --copt=-mfma --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
#    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#RUN pip install /tmp/tensorflow_pkg/tensorflow-1.5.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install tensorflow-gpu==v1.5.0

RUN echo '# LANG' >> ~/.bashrc \
 && echo 'export LANG=ja_JP.UTF-8' >> ~/.bashrc

RUN mkdir -p /root/.vim/bundle \
 && git clone https://github.com/Shougo/neobundle.vim /root/.vim/bundle/neobundle.vim

COPY ./vimrc /root/.vimrc

WORKDIR /root
