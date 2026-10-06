Apptainer Setup
~~~~~~~~~~~~~~~

Converting Docker Image to Apptainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Install Apptainer on your system if it's not already installed (`Installation Guide <https://apptainer.org/docs/user/latest/quick_start.html>`_).

2. Use the ``apptainer pull`` command to convert the Docker image to an Apptainer image:

.. code-block:: console

   $ apptainer pull groundscale.sif docker://milos7250/groundscale:latest

This command will create an Apptainer image named ``groundscale.sif`` by pulling
``milos7250/groundscale:latest`` from Docker Hub.

Building locally
^^^^^^^^^^^^^^^^

To build the image from the repository's definition file instead of pulling a
pre-built image, run from the repository root::

   $ ./docker/build-apptainer.sh

This writes ``docker/groundscale.sif`` using ``docker/apptainer.def``.

Running an Apptainer Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After converting the Docker image to an Apptainer image, you can run GRouNdScale inside the Apptainer container::

   $ apptainer run --nv groundscale.sif --help

To start an interactive shell session within the Apptainer container:

.. code-block:: console

   $ apptainer shell --nv groundscale.sif

* The ``--nv`` flag enables running CUDA application inside the container.

.. warning::
    There might be differences in directory structures and permissions between Apptainer and Docker containers due to Apptainer's bind-mounted approach.
