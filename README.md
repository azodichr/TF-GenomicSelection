# TF-GenomicSelection
Genomic Selection using Neural Networks implemented in TensorFlow


# Setting up virtual environment on HPCC to run TensorFlow

1. To start with, you must use a GPU node
ssh dev-intel16-k80
2. clear default modules
module purge
3. load required modules
module load singularity
module load centos/7.4
 
4. start the Centos container to use the later operating system to
/opt/software/CentOS/7.4--singularity/bin/centos7.4
Note: the prompt will change, create a virtual environment in your home directory
 
module load CUDA/8.0
module load cuDNN/6.0
 
5. create your virtual env for your custom python install.
PYDIR=$HOME/python3-tf
virtualenv --system-site-packages -p python3 $PYDIR

6. activate the new environment
source $PYDIR/bin/activate

7. Install needed packages
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp34-cp34m-linux_x86_64.whl
pip3 install matplotlib
pip3 install numpy
pip3 install pandas


# Set up to run at WISC via GLBRC crediential

1. Install GLBRC VPN for UW. 
https://intranet.wei.wisc.edu/information_technology/Lists/KB/DispForm.aspx?ID=51

2. 
