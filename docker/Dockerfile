#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
# TODO: use FROM statements to get decent building containers
FROM nvcr.io/nvidia/pytorch:24.12-py3

# === Configure environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends \
    openssh-server \ 
    sudo \
    rsync \
    wget \
    bzip2 zip unzip \
    meson


# vips specific dependencies
RUN apt update --fix-missing && \ 
    apt install -y --no-install-recommends \
    libarchive-dev \ 
    libniftiio-dev \
    libglib2.0-dev \ 
    libfftw3-dev \
    libopenexr-dev \
    libgsf-1-dev \
    libglib2.0-dev \
    liborc-dev \
    libopenslide-dev \
    libmatio-dev \
    libwebp-dev \
    libjpeg-turbo8-dev \
    libexpat1-dev \
    libexif-dev \
    libtiff5-dev \
    libcfitsio-dev \
    libpoppler-glib-dev \
    librsvg2-dev \
    libpango1.0-dev \
    libopenjp2-7-dev \
    liblcms2-dev \
    libimagequant-dev


#COPY ./lightstream /tmp/lightstream

# openslide specific dependencies.
# Check libdicom? openslide it's build, but might be useful for vipsdisp
RUN apt update --fix-missing && \ 
    apt install -y --no-install-recommends \
    libsqlite3-dev \
    doxygen \
    valgrind 


WORKDIR /tmp

# Build openslide 4
run wget https://github.com/openslide/openslide/releases/download/v4.0.0/openslide-4.0.0.tar.xz && \
    tar xf openslide-4.0.0.tar.xz && \
    cd openslide-4.0.0 && \
    meson setup build && \
    meson compile -C build && \
    meson install -C build

WORKDIR /tmp

# Build vips 8.16
run wget https://github.com/libvips/libvips/releases/download/v8.16.0/vips-8.16.0.tar.xz && \
    tar xf vips-8.16.0.tar.xz && \
    cd vips-8.16.0 && \
    meson setup build && \
    cd build && \
    meson compile && \ 
    meson test && \
    meson install

RUN pip3 install --upgrade pip && \
    pip3 install lightning==2.3.3&& \
    pip3 install pytorch-lightning==2.3.3&& \
    pip3 install dataclasses_json && \
    pip3 install lightstream && \
    pip3 install albumentationsxl==0.1.2&& \
    pip3 install pyvips && \
    pip3 install wandb && \
    pip3 install pytest
    
# Remove, otherwise hostname is not availble on jupyter lab start
RUN rm /usr/local/etc/jupyter/jupyter_notebook_config.py  


RUN rm -rf /tmp/*

ENTRYPOINT ["/bin/bash"]