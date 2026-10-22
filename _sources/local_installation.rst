Local Installation
~~~~~~~~~~~~~~~~~~

**Prerequisites:** Install Conda and ensure that CUDA drivers are available if
you plan to train on a GPU. The supported environment uses Python 3.11 and
PyTorch 2.9.1.

1. Clone the GRouNdScale repository to a directory of your choice::

   $ git clone https://github.com/milos7250/GRouNdScale.git
   
   .. tip::
       You can optionally clone the scGAN, BEELINE, scDESIGN2, and SPARSim submodules to also get the specific version of repositories that we used in our study. 
        
       .. code-block:: sh
        
           git clone --recurse-submodules https://github.com/milos7250/GRouNdScale.git
           
2. Navigate to the project directory::

   $ cd GRouNdScale

3. Create the supported Conda environment and install the dependencies::

   $ ./conda-env-create.sh

The setup script creates the ``groundscale`` environment from
``environment.yml``, installs PyTorch and the first dependency group from
``requirements.txt``, installs build-sensitive dependencies from
``requirements2.txt``, and applies the required ``arboreto.patch``. It asks
whether developer dependencies from ``requirements-dev.txt`` should also be
installed.

4. Activate the environment::

   $ conda activate groundscale

You're now ready to use GRouNdScale locally. For CPU-only installations, edit
the PyTorch installation command in ``conda-env-create.sh`` to use the CPU
index instead of the CUDA 13 index.

.. admonition:: Troubleshooting

   **Known Installation Issues**

   If you’re encountering issues with installation to ubuntu, it might be because you are missing one or more of the following packages:

   .. code-block:: sh

      sudo apt install build-essential libffi-dev     zlib1g-dev     libncurses5-dev     libgdbm-dev     libnss3-dev     libssl-dev     libreadline-dev     libsqlite3-dev     libpng-dev     libjpeg-dev     libbz2-dev  liblzma-dev tk-dev
