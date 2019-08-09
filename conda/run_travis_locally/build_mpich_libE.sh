# Conda build of dependencies using mpich.
# Source script to maintain environment after running. Script stops if installs fail.
# Note for other MPIs may need to install some packages from source (eg. petsc)

# -x echo commands
set -x # problem with this - loads of conda internal commands shown - overwhelming.

export PYTHON_VERSION=3.6       # default - override with -p <version>
export LIBE_BRANCH="experimental/balsam-on-travis"    # default - override with -b <branchname>
export SYSTEM="Linux"         # default - override with -s <OS>
                                # Options for miniconda - Linux, MacOSX, Windows
export MPI=MPICH
export HYDRA_LAUNCHER=fork

# Allow user to optionally set python version and branch
# E.g: ". ./build_mpich_libE.sh -p 3.4 -b feature/myfeature"
while getopts ":p:b:s:" opt; do
  case $opt in
    p)
      echo "Parameter supplied for Python version: $OPTARG" >&2
      PYTHON_VERSION=$OPTARG
      ;;
    b)
      echo "Parameter supplied for branch name: $OPTARG" >&2
      LIBE_BRANCH=${OPTARG}
      ;;
    s)
      echo "Parameter supplied for OS: $OPTARG" >&2
      SYSTEM=${OPTARG}
      ;;
    \?)
      echo "Invalid option supplied: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

echo -e "\nBuilding libE on ${SYSTEM} with python $PYTHON_VERSION and branch ${LIBE_BRANCH}\n"

# mkdir pg; cd pg;
# wget http://sbp.enterprisedb.com/getfile.jsp?fileid=11966
# tar xf getfile.jsp?fileid=11966
# export PATH="$PATH:/home/travis/pg/pgsql/bin"
# cd ~

sudo pip install --upgrade pip
sudo /etc/init.d/postgresql stop 9.2
sudo /etc/init.d/postgresql start 9.6
export PATH=$PATH:/usr/lib/postgresql/9.6/bin

# This works if not sourced but if sourced its no good.
# set -e

# sudo apt install gfortran || return
# sudo apt install libblas-dev || return
# sudo apt-get install liblapack-dev || return

wget https://repo.continuum.io/miniconda/Miniconda3-latest-${SYSTEM}-x86_64.sh -O miniconda.sh || return
bash miniconda.sh -b -p $HOME/miniconda || return
export PATH="$HOME/miniconda/bin:$PATH" || return
conda update -q -y  conda

conda config --add channels conda-forge || return
conda config --set always_yes yes --set changeps1 no || return
conda create --yes --name condaenv python=$PYTHON_VERSION || return

source activate condaenv || return
wait

# Test to see if this works on MacOS Travis testing environment
if [[ "$SYSTEM" == "MacOSX" ]]; then
  conda install clang_osx-64 || return
else
  conda install gcc_linux-64 || return
fi
conda install nlopt petsc4py petsc mumps-mpi=5.1.2=h5bebb2f_1007 mpi4py scipy $MPI
#conda install numpy || return #scipy includes numpy
# conda install scipy || return
# conda install mpi4py || return
# conda install petsc4py petsc || return
# conda install nlopt || return

# pip install these as the conda installs downgrade pytest on python3.4
pip install balsam-flow || return
pip install pytest || return
pip install pytest-cov || return
pip install pytest-timeout || return
pip install mock || return
pip install coveralls || return

# Not required on travis
git clone -b $LIBE_BRANCH https://github.com/Libensemble/libensemble.git || return
cd libensemble/ || return
pip install -e . || return

export BALSAM_DB_PATH='~/test-balsam'
sudo chown -R postgres:postgres /var/run/postgresql
sudo chmod a+w /var/run/postgresql
balsam init ~/test-balsam
sudo chmod -R 700 ~/test-balsam/balsamdb
source balsamactivate test-balsam

export EXE=$PWD/libensemble/tests/regression_tests/script_test_balsam.py
export NUM_WORKERS=2
export WORKFLOW_NAME=libe_test-balsam
#export SCRIPT_ARGS=$NUM_WORKERS
export LIBE_WALLCLOCK=3
export THIS_DIR=$PWD
export SCRIPT_BASENAME=${EXE%.*}

balsam rm apps --all --force
balsam rm jobs --all --force

balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

MYHOME=$PWD
cd libensemble/tests/regression_tests
python test_balsam.py
cd $MYHOME
#libensemble/tests/run-tests.sh

echo -e "\n\nScript completed...\n\n"
set +ex
