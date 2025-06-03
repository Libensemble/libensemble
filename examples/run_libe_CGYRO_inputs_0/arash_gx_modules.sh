export GACODE_PLATFORM=PERLMUTTER_GPU
#export GACODE_ROOT=/global/cfs/cdirs/atom/atom-install-perlmutter/gacode-gpu
#export GACODE_ROOT=/global/common/software/atom/gacode-perlmutter-gpu
export GACODE_ROOT=/global/cfs/cdirs/m4493/ebelli/gacode
. ${GACODE_ROOT}/shared/bin/gacode_setup
. ${GACODE_ROOT}/platform/env/env.$GACODE_PLATFORM
# export ATOM=/global/cfs/cdirs/atom
# if [ "$GACODE_PLATFORM" == "" ] ; then
#   export GACODE_ROOT=$ATOM/atom-install-cori/gacode-source-mkl
#   export GACODE_PLATFORM=CORI_KNL_HT2_MKL
# #   export GACODE_ROOT=$ATOM/atom-install-cori/gacode-source-mkl
# #   export GACODE_PLATFORM=CORI_KNL_HT2_MKL
#  #  export GACODE_ROOT=$ATOM/atom-install-cori/gacode-source-mklht
#  #  export GACODE_PLATFORM=CORI_KNL_HT2_MKLHT
#   . ${GACODE_ROOT}/shared/bin/gacode_setup
#   . ${GACODE_ROOT}/platform/env/env.$GACODE_PLATFORM
# fi
