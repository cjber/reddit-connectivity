FROM docker.io/cjber/cuda:0.1

ENV PYTHON_VERSION=3.10.8
ENV POETRY_VERSION=1.1.15

RUN reflector --latest 5 --sort rate --save /etc/pacman.d/mirrorlist \
    && pacman -Syu pyenv blas lapack gcc-fortran --noconfirm


ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
ENV DATA_DIR=/reddit-connectivity/data
ENV TOKENIZERS_PARALLELISM=false

RUN pyenv install "${PYTHON_VERSION}" \
    && pyenv global "${PYTHON_VERSION}" \
    && pyenv rehash

USER nobody
WORKDIR $HOME/reddit-connectivity

COPY requirements.txt requirements.txt
COPY src/ ./src

RUN pip install -r requirements.txt --no-cache-dir
