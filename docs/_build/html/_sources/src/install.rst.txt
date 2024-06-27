.. _install:

Installation
============

Introduction
------------

This document provides instructions for installing the DAS4Whales package on your system.

Prerequisites
-------------

Before proceeding with the installation, make sure you have the following prerequisites:

- Python 3.6 or higher
- Pip package manager
- Virtual environment (optional but recommended)

Installation Steps
------------------

Follow these steps to install DAS4Whales:

1. Create a virtual environment (optional but recommended):

    .. code-block:: bash

        python3 -m venv myenv

2. Activate the virtual environment:

    .. code-block:: bash
    
        source myenv/bin/activate
    

3. Install DAS4Whales using pip:

    .. code-block:: bash
    
        pip install 'git+https://github.com/qgoestch/DAS4Whales'

4. Verify the installation:

    .. code-block:: bash

        pip list | grep das4whales

    This command should display the version number of DAS4Whales.

Usage
-----

To use DAS4Whales, refer to the documentation and examples provided in the :doc:`tutorial<tutorial>`

Conclusion
----------

Congratulations! You have successfully installed DAS4Whales on your system. Enjoy using the package!
