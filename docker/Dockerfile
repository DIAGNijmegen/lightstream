
FROM nvcr.io/nvidia/pytorch:23.09-py3
ARG VIPS_VERSION=8.14.5

LABEL maintainer="Stephan"

# === Configure environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        openssh-server \
        sudo \
        meson \
        rsync \
        wget \
        bzip2 zip unzip \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev libjpeg-turbo8\
        libgl1-mesa-glx libgirepository1.0-dev libexif-dev librsvg2-dev fftw3-dev orc-0.4-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd

RUN wget https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz -P /tmp && \
    tar -xf /tmp/vips-${VIPS_VERSION}.tar.xz --directory /tmp/ && \
    rm -rf /tmp/vips-${VIPS_VERSION}.tar.xz && \
    cd /tmp/vips-${VIPS_VERSION} && \
    meson setup build --buildtype release --prefix=/usr/local &&\
    cd build && \
    meson compile && \
    meson test && \
    meson install &&\
    ldconfig && \
    mkdir -p /opt/tests/vips_tests && mv /tmp/vips-${VIPS_VERSION}/test/test-suite /opt/tests/vips_tests &&\
    rm -rf /tmp/vips-${VIPS_VERSION}

RUN pip3 install lightning pyvips

ENTRYPOINT ["/bin/bash"]
