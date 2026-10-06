Docker Setup 
~~~~~~~~~~~~
**Prerequisite:** Before you begin, make sure you have Docker installed on your machine. You can download and install Docker from the official website: `Get Started | Docker <https://www.docker.com/get-started/>`_

Option A: Using Pre-built Docker Image (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Pull the Docker image from Docker Hub::

   $ docker pull milos7250/groundscale:latest

2. Run GRouNdScale inside the Docker container and pass GPU devices::

   $ docker run --gpus all milos7250/groundscale:latest

* The ``--gpus all`` flag enables GPU support within the container. Omit if you intend to use CPU only.

Alternatively, you can run the container with the ``--it`` flag to launch an interactive terminal session::

   $ docker run -it --gpus all milos7250/groundscale:latest /bin/bash

You're now inside the Docker container with CUDA support, ready to use GRouNdScale!

Option B: Building Docker Image from Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Clone the GRouNdScale repository::

   $ git clone https://github.com/milos7250/GRouNdScale.git

2. Navigate to the project directory::

   $ cd GRouNdScale

3. Build the Docker image using the provided build script::
   
   $ ./docker/build-docker.sh

   This builds the local ``groundscale`` image from ``docker/Dockerfile``. The
   image is based on PyTorch 2.14.0 with CUDA 13.2 and Python 3.12.

   .. note::
        
      Building the image using this method may take approximately 15-30 minutes, depending on your system's performance.

4. Run the Docker container and pass GPU devices::

   $ docker run -itd --name yourusername/groundscale:custom --gpus all groundscale /bin/bash

Verifying GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^
Inside the Docker container, you can verify if the GPU is recognized using::

$ nvidia-smi

You should see detailed information about your GPU, including its name, memory usage, etc. This confirms that GPU acceleration is enabled inside the container.

.. code-block:: bash

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.63.01    Driver Version: 470.63.01    CUDA Version: 11.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
    | N/A   33C    P0    41W / 300W |      0MiB / 32480MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
