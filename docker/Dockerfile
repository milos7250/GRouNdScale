FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime AS runtime

# Upgrade system packages
# Also fix cmake wrapper bug
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    rm -rf /usr/local/bin/cmake \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y git g++ make cmake python3-venv

# Install dependencies into an isolated environment using the system Python.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /groundscale

# Install dependencies from requirements.txt
COPY requirements.txt /groundscale/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install \
    --root-user-action ignore \
    -r /groundscale/requirements.txt

COPY requirements2.txt /groundscale/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install \
    --root-user-action ignore \
    --no-build-isolation \
    -r /groundscale/requirements2.txt

# Install development dependencies from requirements-dev.txt (optional, uncomment if needed)
# COPY requirements-dev.txt /groundscale/
# RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --root-user-action ignore -r /groundscale/requirements-dev.txt

# Clean up pip cache to reduce image size
RUN rm -rf /root/.cache/pip || true

# Patch arboreto bug
COPY arboreto.patch /groundscale
RUN patch -d "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/arboreto" < /groundscale/arboreto.patch



FROM runtime AS docs

COPY requirements-dev.txt /groundscale/

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
    --root-user-action ignore \
    -r /groundscale/requirements-dev.txt

WORKDIR /groundscale/
COPY src /groundscale/src
COPY docs /groundscale/docs

ENTRYPOINT ["/opt/venv/bin/sphinx-build"]
CMD ["docs", "docs/_build"]


FROM runtime AS final

WORKDIR /groundscale/
COPY src /groundscale/src

# Run the main Python script; runtime arguments replace the default help flag.
ENTRYPOINT ["/opt/venv/bin/python", "/groundscale/src/main.py"]
CMD ["--help"]
