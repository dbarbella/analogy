New instructions - Tested with Python 3.7.4
* Install Torch as directed here: https://pytorch.org/get-started/locally/
* * As of 8.25.19, this requires Python 3.6.1 or newer.
* * You may need to use py -3.7 -m pip install... instead of whatever it gives you at the link above if you’re using Windows and need to install it specifically for a specific version of Python.
* pip install stanfordnlp
* * Documentation here: https://stanfordnlp.github.io/stanfordnlp/
* Run stanford_nlp_setup.py. If it asks where you want to put the files, say LOC\analogy\2018\stanfordnlp_resources, where LOC is the path to your analogy directory on whatever machine you’re on.
* Download Stanford CoreNLP and models for the language you wish to use.
* Put the model jars in the distribution folder
* * To use python code that uses this, your system needs to know where Stanford CoreNLP is located:
* * export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05
* * * Replace this with the actual path
* Troubleshooting “Timed out waiting for service to come alive”
* * Some IDEs can interfere with things that use this. Run this from a command prompt if necessary.
* * This is also the message it gives when it can’t find coreNLP
* * This can also be the result of trying to allocate more memory than is available.
