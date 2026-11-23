Installation
============


.. Install using pip
.. -----------------

.. .. code-block:: bash

..    pip install my-package

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/rferrerc/pyflip.git@main

   cd pyflip

   pip install .

pyFLIP builds its JWST models based off of STPSF, a Python library to simulate JWST PSFs. See installation instructions in the STPSF documentation (importantly: including the required data files, and the required setting of environment variables).