#!/usr/bin/env bash

# Libensemble Test Runner

# If hooks/set-hooks.sh is run - this runs as a pre-push git script

# Options for test types (only matters if "true" or anything else)
export RUN_UNIT_TESTS=true    #Recommended for pre-push / CI tests
export RUN_COV_TESTS=true     #Provide coverage report
export RUN_REG_TESTS=true     #Recommended for pre-push / CI tests
export RUN_PEP_TESTS=false    #Code style conventions
export PYTHON_FLAGS=''        #Flags for PYTHON_RUN

# Regression test options
#export REG_TEST_LIST='test_number1.py test_number2.py' #selected/ordered
export REG_TEST_LIST=test_*.py #unordered # override with -y
# export REG_TEST_PROCESS_COUNT_LIST='2 4'
export REG_USE_PYTEST=false
export REG_TEST_OUTPUT_EXT=std.out #/dev/null
export REG_STOP_ON_FAILURE=false
export REG_RUN_LARGEST_TEST_ONLY=true
export RUN_EXTRA=false
#-----------------------------------------------------------------------------------------

# Test Directories - all relative to project root dir
export CODE_DIR=libensemble
export LIBE_SRC_DIR=$CODE_DIR
export TESTING_DIR=$CODE_DIR/tests
export UNIT_TEST_SUBDIR=$TESTING_DIR/unit_tests
export UNIT_TEST_NOMPI_SUBDIR=$TESTING_DIR/unit_tests_nompi
export UNIT_TEST_LOGGER_SUBDIR=$TESTING_DIR/unit_tests_logger
export REG_TEST_SUBDIR=$TESTING_DIR/regression_tests
export FUNC_TEST_SUBDIR=$TESTING_DIR/functionality_tests

#Coverage merge and report dir - will need the relevant .coveragerc file present
#export COV_MERGE_DIR='' #root dir
export COV_MERGE_DIR=$TESTING_DIR

#PEP code standards test options
export PYTHON_PEP_STANDARD=pep8

#export PEP_SCOPE=$CODE_DIR
export PEP_SCOPE=$LIBE_SRC_DIR

#-----------------------------------------------------------------------------------------
#Functions

#Does file exist in given dir
#In:  Directory path, filename
#Out: true if found, else false
find_file_in_dir () {
  local xdir=$1
  local filename=$2
  file_found_in_tree=false
  fout=$(find "$xdir" -maxdepth 1 -name $filename)
  if [[ ${fout##*/} == "$filename" ]]; then
    file_found_in_tree=true
  fi;
  echo "$file_found_in_tree"
}

#Print summary line like pytest
#In:  String to print
#Out: Prints line (nothing returned)
print_summary_line() {
  local phrase
  phrase=$@
  #Add two for the spaces
  textsize=$((${#phrase}+2))
  #echo $textsize
  tsize=$(tput cols)
  lsize=$((($tsize-$textsize)/2))

  #Deal with result of integer division
  tot_len=$((${lsize}+${lsize}+${textsize}))
  shortfall=$(($tsize-$tot_len))

  symbol_count=$lsize
  while [ $symbol_count -gt 0 ]
  do
    printf "="
    symbol_count=$((symbol_count-1))
  done
  printf " $phrase "
  #printf '%*s' "$phrase"
  #echo '$phrase'
  symbol_count=$(($lsize+$shortfall))
  while [ $symbol_count -gt 0 ]
  do
    printf "="
    symbol_count=$((symbol_count-1))
  done
}

#Return a time difference
#In:  Start and End times as strings
#Out: Time difference as a string
total_time() {
  #Is bc present
  USE_BC=f
  bc --version >> /dev/null && USE_BC=t
  if [ $USE_BC = 't' ]; then
    diff=$(echo "scale=2;($2 - $1)*100/100" | bc)
  else
    diff=$(( $2 - $1 ))
  fi;
  echo "$diff"
}

#Cleanup test run directories
cleanup() {
  THISDIR=${PWD}
  cd $ROOT_DIR/$TESTING_DIR
    filelist=(.cov_merge_out*);        [ -e ${filelist[0]} ] && rm .cov_merge_out*
    filelist=(ensemble_*);             [ -e ${filelist[0]} ] && rm -r ensemble_*
  for DIR in $UNIT_TEST_SUBDIR $UNIT_TEST_NOMPI_SUBDIR $UNIT_TEST_LOGGER_SUBDIR ; do
  cd $ROOT_DIR/$DIR
    filelist=(libE_history_at_abort_*.npy); [ -e ${filelist[0]} ] && rm libE_history_at_abort_*.npy
    filelist=(*.out);                   [ -e ${filelist[0]} ] && rm *.out
    filelist=(*.err);                   [ -e ${filelist[0]} ] && rm *.err
    filelist=(*.pickle);                [ -e ${filelist[0]} ] && rm *.pickle
    filelist=(.cov_unit_out*);          [ -e ${filelist[0]} ] && rm .cov_unit_out*
    filelist=(simdir/*.x);              [ -e ${filelist[0]} ] && rm simdir/*.x
    filelist=(libe_task_*.out);         [ -e ${filelist[0]} ] && rm libe_task_*.out
    filelist=(*libE_stats.txt);         [ -e ${filelist[0]} ] && rm *libE_stats.txt
    filelist=(my_machinefile);          [ -e ${filelist[0]} ] && rm my_machinefile
    filelist=(libe_stat_files);         [ -e ${filelist[0]} ] && rm -r libe_stat_files
    filelist=(ensemble.log);            [ -e ${filelist[0]} ] && rm ensemble.log
    filelist=(H_test.npy);              [ -e ${filelist[0]} ] && rm H_test.npy
  done
  for DIR in $REG_TEST_SUBDIR $FUNC_TEST_SUBDIR; do
  cd $ROOT_DIR/$DIR
    filelist=(*.$REG_TEST_OUTPUT_EXT);  [ -e ${filelist[0]} ] && rm *.$REG_TEST_OUTPUT_EXT
    filelist=(*.npy);                   [ -e ${filelist[0]} ] && rm *.npy
    filelist=(*.pickle);                [ -e ${filelist[0]} ] && rm *.pickle
    filelist=(.cov_reg_out*);           [ -e ${filelist[0]} ] && rm .cov_reg_out*
    filelist=(*active_runs.txt);        [ -e ${filelist[0]} ] && rm *active_runs.txt
    filelist=(*.err);                   [ -e ${filelist[0]} ] && rm *.err
    filelist=(outfile*.txt);            [ -e ${filelist[0]} ] && rm outfile*.txt
    filelist=(machinefile*);            [ -e ${filelist[0]} ] && rm machinefile*
    filelist=(libe_task_*.out);         [ -e ${filelist[0]} ] && rm libe_task_*.out
    filelist=(*libE_stats.txt);         [ -e ${filelist[0]} ] && rm *libE_stats.txt
    filelist=(my_simtask.x);            [ -e ${filelist[0]} ] && rm my_simtask.x
    filelist=(libe_stat_files);         [ -e ${filelist[0]} ] && rm -r libe_stat_files
    filelist=(ensemble.log);            [ -e ${filelist[0]} ] && rm ensemble.log
    filelist=(ensemble_*);              [ -e ${filelist[0]} ] && rm -r ensemble_*
    filelist=(sim_*);                   [ -e ${filelist[0]} ] && rm -r sim_*
    filelist=(gen_*);                   [ -e ${filelist[0]} ] && rm -r gen_*
    filelist=(nodelist_*);              [ -e ${filelist[0]} ] && rm nodelist_*
    filelist=(x_*.txt y_*.txt);         [ -e ${filelist[0]} ] && rm x_*.txt y_*.txt
    filelist=(opt_*.txt_flag);          [ -e ${filelist[0]} ] && rm opt_*.txt_flag
    filelist=(*.png);                   [ -e ${filelist[0]} ] && rm *.png
  done
  cd $THISDIR
}

#-----------------------------------------------------------------------------------------

#Parse Options
#set -x

unset PYTHON_VER
unset RUN_PREFIX

#Default to script name for run-prefix (name of tests)
script_name=`basename "$0"`
RUN_PREFIX=$script_name
CLEAN_ONLY=false
unset MPIEXEC_FLAGS
PYTEST_SHOW_OUT_ERR=false
RTEST_SHOW_OUT_ERR=false

export RUN_MPI=false
export RUN_LOCAL=false
export RUN_TCP=false

usage() {
  echo -e "\nUsage:"
  echo "  $0 [-hcszurmlte] [-p <2|3>] [-A <string>] [-n <string>] [-a <string>] [-y <string>]" 1>&2;
  echo ""
  echo "Options:"
  echo "  -h              Show this help message and exit"
  echo "  -c              Clean up test directories and exit"
  echo "  -s              Print stdout and stderr to screen when running pytest (unit tests)"
  echo "  -z              Print stdout and stderr to screen when running regression tests (run without pytest)"
  echo "  -u              Run only the unit tests"
  echo "  -r              Run only the regression tests"
  echo "  -m              Run the regression tests using MPI comms"
  echo "  -l              Run the regression tests using Local comms"
  echo "  -t              Run the regression tests using TCP comms"
  echo "  -e              Run extra unit and regression tests that require additional dependencies"
  echo "  -p {version}    Select a version of python. E.g. -p 2 will run with the python2 exe"
  echo "                  Note: This will literally run the python2/python3 exe. Default runs python"
  echo "  -A {-flag arg}  Supply arguments to python"
  echo "  -n {name}       Supply a name to this test run"
  echo "  -a {args}       Supply a string of args to add to mpiexec line"
  echo "  -y {args}       Supply a list of regression tests as a reg. expression  e.g. '-y test_persistent_aposmm*'"
  echo ""
  echo "Note: If none of [-mlt] are given, the default is to run tests for all comms"
  echo ""
  exit 1
}

while getopts ":p:n:a:y:A:hcszurmlte" opt; do
  case $opt in
    p)
      echo "Parameter supplied for Python version: $OPTARG" >&2
      PYTHON_VER=$OPTARG
      ;;
    n)
      echo "Parameter supplied for Test Name: $OPTARG" >&2
      RUN_PREFIX=$OPTARG
      ;;
    a)
      echo "Parameter supplied for mpiexec args: $OPTARG" >&2
      MPIEXEC_FLAGS=$OPTARG
      ;;
    c)
      #echo "Cleaning test output: $OPTARG" >&2
      echo "Cleaning test output"
      CLEAN_ONLY=true
      ;;
    s)
      echo "Will show stdout and stderr during pytest"
      PYTEST_SHOW_OUT_ERR=true
      ;;
    z)
      echo "Will show stdout and stderr during regression tests"
      RTEST_SHOW_OUT_ERR=true
      ;;    u)
      echo "Running only the unit tests"
      export RUN_REG_TESTS=false
      ;;
    r)
      echo "Running only the regression tests"
      export RUN_UNIT_TESTS=false
      ;;
    l)
      echo "Running only the local regression tests"
      export RUN_LOCAL=true
      ;;
    t)
      echo "Running only the TCP regression tests"
      export RUN_TCP=true
      ;;
    e)
      echo "Running extra tests with additional dependencies"
      export RUN_EXTRA=true
      ;;
    m)
      echo "Running only the MPI regression tests"
      export RUN_MPI=true
      ;;
    y)
      echo "Running with user supplied test list"
      export REG_TEST_LIST="$OPTARG"
      ;;
    A)
      echo "Python arguments passed: $OPTARG" >&2
      PYTHON_FLAGS="$PYTHON_FLAGS $OPTARG"
      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option supplied: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
# shift $((OPTIND-1))
# if [ -z "${s}" ] || [ -z "${p}" ]; then
#     usage
# fi

# If none selected default to running all tests
if [ "$RUN_MPI" = false ] && [ "$RUN_LOCAL" = false ] && [ "$RUN_TCP" = false ]; then
    RUN_MPI=true && RUN_LOCAL=true && RUN_TCP=false
fi

#-----------------------------------------------------------------------------------------

# Get project root dir

# Try using git root dir
root_found=false
ROOT_DIR=$(git rev-parse --show-toplevel) && root_found=true

#If not found using git - try search up tree for setup.py
if [[ $root_found == "false" ]]; then
  search_dir=`pwd`
  search_for="setup.py"
  while [ "$search_dir" != "/" ]
  do
    file_found=$(find_file_in_dir $search_dir $search_for)
    #[[ $file_found = "true" ]] && break
    if [[ $file_found = "true" ]]; then
      ROOT_DIR=$search_dir
      root_found=true
      break;
    fi;
    search_dir=`dirname "$search_dir"`
  done
fi;

if [ $CLEAN_ONLY = "true" ]; then
  if [ "$root_found" = true ]; then
    cleanup
  else
    echo -e "Clean up aborted - no project root directory found"
  fi;
  exit
fi;

#If not supplied will go to just python (no number) - eg. with tox/virtual envs
PYTHON_RUN="python$PYTHON_VER $PYTHON_FLAGS"
echo -e "Python run: $PYTHON_RUN"

#Get current time in seconds
#In:  Nothing
#Out: Returns time in seconds (seconds since 1970-01-01 00:00:00 UTC) as a string
#     Or if bc not available uses SECONDS (whole seconds that script has been running)
current_time() {
  local time
  #Is bc present
  USE_BC=f
  bc --version >> /dev/null && USE_BC=t
  if [ $USE_BC = 't' ]; then
    #time=$(date +%s.%N)
    time=$($PYTHON_RUN -c 'import time; print(time.time())')
  else
    time=$SECONDS
  fi;
  echo "$time"
}

textreset=$(tput sgr0)
fail_color=$(tput bold; tput setaf 1) #red
pass_color=$(tput bold; tput setaf 2) #green
titl_color=$(tput bold; tput setaf 6) #cyan
hint_color=$(tput bold; tput setaf 4) #blue

# Note - pytest exit codes
# Exit code 0:  All tests were collected and passed successfully
# Exit code 1:  Tests were collected and run but some of the tests failed
# Exit code 2:  Test execution was interrupted by the user
# Exit code 3:  Internal error happened while executing tests
# Exit code 4:  pytest command line usage error
# Exit code 5:  No tests were collected

tput bold
#echo -e "\nRunning $RUN_PREFIX libEnsemble Test-suite .......\n"
echo -e "\n************** Running: libEnsemble Test-Suite **************\n"
tput sgr 0
echo -e "Selected:"
[ $RUN_UNIT_TESTS = "true" ] && echo -e "Unit Tests"
[ $RUN_REG_TESTS = "true" ]  && echo -e "Regression Tests"
[ $RUN_REG_TESTS = "true" ]  && [ $RUN_MPI = "true" ]   && echo -e " - Run tests with MPI Comms"
[ $RUN_REG_TESTS = "true" ]  && [ $RUN_LOCAL = "true" ] && echo -e " - Run tests with Local Comms"
[ $RUN_REG_TESTS = "true" ]  && [ $RUN_TCP = "true" ]   && echo -e " - Run tests with TCP Comms"
[ $RUN_COV_TESTS = "true" ]  && echo -e "Including coverage analysis"
[ $RUN_PEP_TESTS = "true" ]  && echo -e "PEP Code Standard Tests (static code test)"

COV_LINE_SERIAL=''
COV_LINE_PARALLEL=''
if [ $RUN_COV_TESTS = "true" ]; then
   COV_LINE_SERIAL='--cov --cov-report html:cov_unit'
   #COV_LINE_PARALLEL='-m coverage run --parallel-mode --rcfile=../.coveragerc' #running in sub-dirs
   COV_LINE_PARALLEL='-m coverage run --parallel-mode --concurrency=multiprocessing' #running in regression dir itself

   #include branch coverage? eg. flags if never jumped a statement block... [see .coveragerc file]
   #COV_LINE_PARALLEL='-m coverage run --branch --parallel-mode'
fi;

if [ "$root_found" = true ]; then

  #Running without subdirs - delete any leftover output and coverage data files
  cleanup

  cd $ROOT_DIR/

  # Run Unit Tests -----------------------------------------------------------------------

  if [ "$RUN_UNIT_TESTS" = true ]; then
    tput bold; tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running unit tests"
    tput sgr 0

    if [ "$RUN_EXTRA" = true ]; then
        EXTRA_UNIT_ARG="--runextra"
    fi

    for DIR in $UNIT_TEST_SUBDIR $UNIT_TEST_NOMPI_SUBDIR $UNIT_TEST_LOGGER_SUBDIR ; do
    cd $ROOT_DIR/$DIR

    # unit test subdirs dont contain pytest's conftest.py that defines extra arg
    if [ $DIR = $UNIT_TEST_NOMPI_SUBDIR ]; then
        EXTRA_UNIT_ARG=""
    fi

    if [ "$PYTEST_SHOW_OUT_ERR" = true ]; then
      $PYTHON_RUN -m pytest --capture=no --timeout=120 $COV_LINE_SERIAL $EXTRA_UNIT_ARG #To see std out/err while running
    else
      $PYTHON_RUN -m pytest --timeout=120 $COV_LINE_SERIAL $EXTRA_UNIT_ARG
    fi;

    code=$?
    if [ "$code" -eq "0" ]; then
      echo
      tput bold; tput setaf 2; echo "Unit tests passed. Continuing..."; tput sgr 0
      echo
    else
      echo
      tput bold; tput setaf 1; echo -e "Abort $RUN_PREFIX: Unit tests failed: $code"; tput sgr 0
      exit $code #return pytest exit code
    fi;
    done
  fi;
  cd $ROOT_DIR/

  # Run Regression Tests -----------------------------------------------------------------

  if [ "$RUN_REG_TESTS" = true ]; then
    tput bold; tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running regression tests"
    tput sgr 0

    for DIR in $REG_TEST_SUBDIR $FUNC_TEST_SUBDIR; do
      cd $ROOT_DIR/$DIR

      #Check output dir exists.
      if [ ! -d output ]; then
        mkdir output/
      fi;

    done

    TIMEOUT=""
    if [ -x "$(command -v timeout)" ] ; then
        # TIMEOUT="timeout 60s"
        TIMEOUT=""
    fi
    #Build any sim/gen source code dependencies here .....

    # cd $ROOT_DIR/$REG_TEST_SUBDIR

    #Run regression tests using MPI
    #Before first test set error code to zero
    code=0

    if [ "$REG_USE_PYTEST" = true ]; then
      echo -e "Regression testing using pytest"
      [ $RUN_COV_TESTS = "true" ]  && echo -e "WARNING: Coverage NOT being run for regression tests - not working with pytest\n"
    else
      echo -e "Regression testing is NOT using pytest"
    fi

    echo -e ""

    # ********* Loop over regression tests ************

    reg_start=$(current_time)
    reg_count_tests=0
    reg_count_runs=0
    reg_pass=0
    reg_fail=0
    test_num=0

    for DIR in $REG_TEST_SUBDIR $FUNC_TEST_SUBDIR
    do
      cd $ROOT_DIR/$DIR

      for TEST_SCRIPT in $REG_TEST_LIST
      do
        COMMS_LIST=$(sed -n '/# TESTSUITE_COMMS/s/# TESTSUITE_COMMS: //p' $TEST_SCRIPT)
        IS_EXTRA=$(sed -n '/# TESTSUITE_EXTRA/s/# TESTSUITE_EXTRA: //p' $TEST_SCRIPT)

        if [[ "$IS_EXTRA" = "true" ]] && [[ "$RUN_EXTRA" = false ]]; then
          continue
        fi

        for LAUNCHER in $COMMS_LIST
        do
          #Need proc count here for now - still stop on failure etc.
          NPROCS_LIST=$(sed -n '/# TESTSUITE_NPROCS/s/# TESTSUITE_NPROCS: //p' $TEST_SCRIPT)
          OS_SKIP_LIST=$(sed -n '/# TESTSUITE_OS_SKIP/s/# TESTSUITE_OS_SKIP: //p' $TEST_SCRIPT)
          OMPI_SKIP=$(sed -n '/# TESTSUITE_OMPI_SKIP/s/# TESTSUITE_OMPI_SKIP: //p' $TEST_SCRIPT)
          if [ "$REG_RUN_LARGEST_TEST_ONLY" = true ]; then  NPROCS_LIST=${NPROCS_LIST##*' '}; fi
          for NPROCS in ${NPROCS_LIST}
          do
            NWORKERS=$((NPROCS-1))

            RUN_TEST=false
            if [ "$RUN_MPI" = true ]   && [ "$LAUNCHER" = mpi ];   then RUN_TEST=true; fi
            if [ "$RUN_LOCAL" = true ] && [ "$LAUNCHER" = local ]; then RUN_TEST=true; fi
            if [ "$RUN_TCP" = true ]   && [ "$LAUNCHER" = tcp ];   then RUN_TEST=true; fi

            if [[ "$OSTYPE" = *"darwin"* ]] && [[ "$OS_SKIP_LIST" = *"OSX"* ]]; then
              echo "Skipping test number for OSX: " $test_num
              continue
            fi

            if [[ "$OSTYPE" = *"msys"* ]] && [[ "$OS_SKIP_LIST" = *"WIN"* ]]; then
              echo "Skipping test number for Windows: " $test_num
              continue
            fi

            if [[ "$OMPI_SKIP" = "true" ]] && [[ "$MPIEXEC_FLAGS" = "--oversubscribe" ]] && [[ "$RUN_MPI" = true ]]; then
              echo "Skipping test number for Open MPI: " $test_num
              continue
            fi

            if [ $REG_STOP_ON_FAILURE = "true" ]; then
              #Before Each Test check code is 0 (passed so far) - or skip to test summary
              if [ "$code" -ne "0" ]; then
                RUN_TEST=false
                break
              fi
            fi

            if [ "$RUN_TEST" = "true" ]; then
               test_num=$((test_num+1))
               test_start=$(current_time)

               echo -e "\n ${titl_color}---Test $test_num: $TEST_SCRIPT starting with $LAUNCHER on $NPROCS processes ${textreset}"

               if [ "$REG_USE_PYTEST" = true ]; then
                 if [ "$LAUNCHER" = mpi ]; then
                   mpiexec -np $NPROCS $MPIEXEC_FLAGS $PYTHON_RUN -m pytest $TEST_SCRIPT >> $TEST_SCRIPT.$NPROCS'procs'.$REG_TEST_OUTPUT_EXT 2>test.err
                   test_code=$?
                 else
                   $TIMEOUT $PYTHON_RUN -m pytest $TEST_SCRIPT --comms $LAUNCHER --nworkers $NWORKERS >> $TEST_SCRIPT.$NPROCS'procs'-$LAUNCHER.$REG_TEST_OUTPUT_EXT 2>test.err
                 fi
               else
                 if [ "$LAUNCHER" = mpi ]; then
                   if [ "$RTEST_SHOW_OUT_ERR" = "true" ]; then
                     mpiexec -np $NPROCS $MPIEXEC_FLAGS $PYTHON_RUN $COV_LINE_PARALLEL $TEST_SCRIPT
                     test_code=$?
                   else
                     mpiexec -np $NPROCS $MPIEXEC_FLAGS $PYTHON_RUN $COV_LINE_PARALLEL $TEST_SCRIPT >> $TEST_SCRIPT.$NPROCS'procs'.$REG_TEST_OUTPUT_EXT 2>test.err
                     test_code=$?
                   fi
                 else
                   if [ "$RTEST_SHOW_OUT_ERR" = "true" ]; then
                     $TIMEOUT $PYTHON_RUN $COV_LINE_PARALLEL $TEST_SCRIPT --comms $LAUNCHER --nworkers $NWORKERS
                     test_code=$?
                   else
                     $TIMEOUT $PYTHON_RUN $COV_LINE_PARALLEL $TEST_SCRIPT --comms $LAUNCHER --nworkers $NWORKERS >> $TEST_SCRIPT.$NPROCS'procs'-$LAUNCHER.$REG_TEST_OUTPUT_EXT 2>test.err
                     test_code=$?
                   fi
                 fi
               fi
               reg_count_runs=$((reg_count_runs+1))
               test_end=$(current_time)
               test_time=$(total_time $test_start $test_end)

               if [ "$test_code" -eq "0" ]; then
                 echo -e " ---Test $test_num: $TEST_SCRIPT using $LAUNCHER on $NPROCS processes ${pass_color} ...passed after ${test_time} seconds ${textreset}"
                 reg_pass=$((reg_pass+1))
                 #continue testing
               else
                 echo -e " ---Test $test_num: $TEST_SCRIPT using $LAUNCHER on $NPROCS processes ${fail_color}  ...failed after ${test_time} seconds ${textreset}"
                 code=$test_code #sh - currently stop on failure
                 if [ $REG_STOP_ON_FAILURE != "true" ]; then
                   #Dump error to log file
                   echo -e "\nTest $test_num: $TEST_SCRIPT using $LAUNCHER on $NPROCS processes:\n" >>log.err
                   [ -e test.err ] && cat test.err >>log.err
                 fi;
                 reg_fail=$((reg_fail+1))
               fi;

               #If use sub-dirs - move this test's coverage files to regression dir where they can be merged with other tests
               #[ "$RUN_COV_TESTS" = "true" ] && mv .cov_reg_out.* ../

            fi; #if [ "$RUN_TEST" = "true" ];

          done #nprocs
        done #launcher

        [ $REG_STOP_ON_FAILURE = "true" ] && [ "$code" -ne "0" ] && cat test.err && break
        reg_count_tests=$((reg_count_tests+1))

      done #tests
    done #functionality/regression directories
    reg_end=$(current_time)
    reg_time=$(total_time $reg_start $reg_end)

    # ********* End Loop over regression tests *********

    #Create Coverage Reports ----------------------------------------------

    #Only if passed
    if [ "$code" -eq "0" ]; then

      echo -e "\n..Moving output files to output dir"

      for DIR in $REG_TEST_SUBDIR $FUNC_TEST_SUBDIR
      do
        cd $ROOT_DIR/$DIR

        if [ "$(ls -A output)" ]; then
          rm output/* #Avoid mixing test run results
        fi;

        #sh - shld active_runs be prefixed for each job
        filelist=(*.$REG_TEST_OUTPUT_EXT);   [ -e ${filelist[0]} ] && mv *.$REG_TEST_OUTPUT_EXT output/
        filelist=(*.npy);                    [ -e ${filelist[0]} ] && mv *.npy output/
        filelist=(*active_runs.txt);         [ -e ${filelist[0]} ] && mv *active_runs.txt output/

      done
      cd $ROOT_DIR

      if [ "$RUN_COV_TESTS" = true ]; then

        # Merge MPI coverage data for all ranks from regression tests and create html report in sub-dir

        for DIR in $REG_TEST_SUBDIR $FUNC_TEST_SUBDIR
        do
          cd $ROOT_DIR/$DIR

          # Must combine all if in sep sub-dirs will copy to dir above
          coverage combine .cov_reg_out* #Name of coverage data file must match that in .coveragerc in reg test dir.
          coverage html
          echo -e "..Coverage HTML written to dir $DIR/cov_reg/"

        done
        cd $ROOT_DIR

        if [ "$RUN_UNIT_TESTS" = true ]; then

          #Combine with unit test coverage at top-level
          cd $ROOT_DIR/$COV_MERGE_DIR
          cp $ROOT_DIR/$UNIT_TEST_SUBDIR/.cov_unit_out .
          cp $ROOT_DIR/$UNIT_TEST_NOMPI_SUBDIR/.cov_unit_out2 .
          cp $ROOT_DIR/$UNIT_TEST_LOGGER_SUBDIR/.cov_unit_out3 .
          cp $ROOT_DIR/$REG_TEST_SUBDIR/.cov_reg_out .
          cp $ROOT_DIR/$FUNC_TEST_SUBDIR/.cov_reg_out2 .

          #coverage combine --rcfile=.coverage_merge.rc .cov_unit_out .cov_reg_out
          coverage combine .cov_unit_out .cov_unit_out2 .cov_unit_out3 .cov_reg_out .cov_reg_out2 #Should create .cov_merge_out - see .coveragerc
          coverage html #Should create cov_merge/ dir
          echo -e "..Combined Unit Test/Regression Test Coverage HTML written to dir $COV_MERGE_DIR/cov_merge/"

        else

          # Still need to move reg cov results to COV_MERGE_DIR
          cd $ROOT_DIR/$COV_MERGE_DIR
          cp $ROOT_DIR/$REG_TEST_SUBDIR/.cov_reg_out .
          cp $ROOT_DIR/$FUNC_TEST_SUBDIR/.cov_reg_out2 .

          coverage combine .cov_reg_out .cov_reg_out2
          coverage html
          echo -e "..Combined Regression Test Coverage HTML written to dir $COV_MERGE_DIR/cov_merge/"
        fi;
      fi;
    fi;

    #All reg tests - summary ----------------------------------------------
    if [ "$code" -eq "0" ]; then
      echo
      #tput bold; tput setaf 2

      if [ "$REG_USE_PYTEST" != true ]; then
        #sh - temp formatting similar(ish) to pytest - update in python (as with timing)
        #tput bold; tput setaf 4; echo -e "***Note***: temporary formatting/timing ......"

        summ_line="$reg_pass passed in $reg_time seconds"
        tput bold; tput setaf 2;
        print_summary_line $summ_line
        tput sgr 0
      fi;

      tput bold; tput setaf 2; echo -e "\nRegression tests passed ..."
      tput sgr 0
      echo
    else
      if [ $REG_STOP_ON_FAILURE != "true" ]; then
        echo -e ""
        if [ "$RTEST_SHOW_OUT_ERR" != "true" ]; then
          echo -e "\n..see error log at $REG_TEST_SUBDIR/log.err"
        fi
        summ_line="$reg_fail failed, $reg_pass passed in $reg_time seconds"
        tput bold; tput setaf 1;
        print_summary_line $summ_line
        tput sgr 0
      fi;
      echo
      tput bold; tput setaf 1; echo -e "\nAbort $RUN_PREFIX: Regression tests failed (exit code $code)"; tput sgr 0
      echo
      exit $code
    fi;

  fi; #$RUN_REG_TESTS

  # Run Code standards Tests -----------------------------------------
  cd $ROOT_DIR
  if [ "$RUN_PEP_TESTS" = true ]; then
    tput bold; tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running PEP tests - All python src below $PEP_SCOPE"
    tput sgr 0
    pytest --$PYTHON_PEP_STANDARD $ROOT_DIR/$PEP_SCOPE

    code=$?
    if [ "$code" -eq "0" ]; then
      echo
      tput bold; tput setaf 2; echo "PEP tests passed. Continuing..."; tput sgr 0
      echo
    else
      echo
      tput bold; tput setaf 1; echo -e "Abort $RUN_PREFIX: PEP tests failed: $code"; tput sgr 0
       exit $code #return pytest exit code
    fi;
  fi;

  # ------------------------------------------------------------------
  tput bold; tput setaf 2; echo -e "\n$RUN_PREFIX --$PYTHON_RUN: All tests passed\n"; tput sgr 0
  exit 0
else
  tput bold; tput setaf 1; echo -e "Abort $RUN_PREFIX:  Project root dir not found"; tput sgr 0
  exit 1
fi
