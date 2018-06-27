#theta
export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH

export LD_LIBRARY_PATH=~/.conda/envs/balsam/lib:$LD_LIBRARY_PATH

#My addition - takes non-conda local dirs out of sys path - to help isolate conda
export PYTHONNOUSERSITE=1

. activate balsam #Should be a check here.
