Installation
============

Install via PyPI
----------------

The ``fastlowess`` package can be installed via pip:

.. code-block:: bash

    pip install fastlowess

Install via Conda
-----------------

You can also install from conda-forge:

.. code-block:: bash

    conda install -c conda-forge fastlowess

Building from Source
--------------------

To build from source, you will need Rust installed.

.. code-block:: bash

    maturin develop --release

Verifying Installation
----------------------

You can verify the installation by running a simple Python command:

.. code-block:: python

    import fastlowess
    print(f"fastlowess version: {fastlowess.__version__}")
