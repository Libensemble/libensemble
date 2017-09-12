#!/usr/bin/env bash

# Libensemble Test Runner

# If hooks/set-hooks.sh is run - this runs as a pre-push git script

# Options for test types (only matters if "true" or anything else)
export RUN_UNIT_TESTS=true    #Recommended for pre-push / CI tests
export RUN_COV_TESTS=true     #Provide coverage report
export RUN_REG_TESTS=true     #Recommended for pre-push / CI tests
export RUN_PEP_TESTS=false     #Code syle conventions

# Regression test options
#export REG_TEST_LIST='test_libE_on_GKLS_aposmm_1.py test_number2.py' #selected/ordered
export REG_TEST_LIST=test_*.py #unordered
export REG_TEST_PROCESS_COUNT_LIST='2 4'
export REG_USE_PYTEST=false
export REG_TEST_OUTPUT_EXT=std.out #/dev/null
export REG_STOP_ON_FAILURE=false
#-----------------------------------------------------------------------------------------
     
# Test Directories - all relative to project root dir
export CODE_DIR=code
export LIBE_SRC_DIR=$CODE_DIR/src
export TESTING_DIR=$CODE_DIR/tests
export UNIT_TEST_SUBDIR=$TESTING_DIR/unit_tests
export REG_TEST_SUBDIR=$TESTING_DIR/regression_tests
export GKLS_BUILD_DIR=$CODE_DIR/examples/sim_funcs/GKLS/GKLS_sim_src

#Coverage merge and report dir - will need the relevant .coveragerc file present
#export COV_MERGE_DIR='' #root dir
export COV_MERGE_DIR=$TESTING_DIR

#PEP code standards test options
export PYTHON_PEP_STANDARD=pep8

#export PEP_SCOPE=$CODE_DIR
export PEP_SCOPE=$LIBE_SRC_DIR

#-----------------------------------------------------------------------------------------
#Functions

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
    time=$(date +%s.%N)
  else
    time=$SECONDS
  fi;
  echo "$time"
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

#Cleanup - esp regression test run directory
#Changes dirs
cleanup() {
  cd $ROOT_DIR/$REG_TEST_SUBDIR
    filelist=(*.$REG_TEST_OUTPUT_EXT); [ -e ${filelist[0]} ] && rm *.$REG_TEST_OUTPUT_EXT
    filelist=(*.npy);                  [ -e ${filelist[0]} ] && rm *.npy
    filelist=(.cov_reg_out*);          [ -e ${filelist[0]} ] && rm .cov_reg_out*
    filelist=(*active_runs.txt);       [ -e ${filelist[0]} ] && rm *active_runs.txt
    filelist=(*.err);                  [ -e ${filelist[0]} ] && rm *.err
    filelist=(outfile*.txt);           [ -e ${filelist[0]} ] && rm outfile*.txt
    filelist=(machinefile*);           [ -e ${filelist[0]} ] && rm machinefile*
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

while getopts ":p:n:c" opt; do
  case $opt in
    p)
      echo "Parameter supplied for Python version: $OPTARG" >&2
      PYTHON_VER=$OPTARG
      ;;
    n)
      echo "Parameter supplied for Test Name: $OPTARG" >&2
      RUN_PREFIX=$OPTARG
      ;;
    c)
      #echo "Cleaning test output: $OPTARG" >&2
      echo "Cleaning test output"
      CLEAN_ONLY=true
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

#-----------------------------------------------------------------------------------------

# Using git root dir
root_found=false
ROOT_DIR=$(git rev-parse --show-toplevel) && root_found=true

if [ $CLEAN_ONLY = "true" ]; then
 cleanup
 exit
fi;

#If not supplied will go to just python (no number) - eg. with tox/virtual envs
PYTHON_RUN=python$PYTHON_VER
echo -e "Python run: $PYTHON_RUN"

textreset=$(tput sgr0)
fail_color=$(tput bold;tput setaf 1) #red
pass_color=$(tput bold;tput setaf 2) #green
titl_colour=$(tput bold;tput setaf 6) #cyan
hint_colour=$(tput bold;tput setaf 4) #blue

# Note - pytest exit codes
# Exit code 0:  All tests were collected and passed successfully
# Exit code 1:  Tests were collected and run but some of the tests failed
# Exit code 2:  Test execution was interrupted by the user
# Exit code 3:  Internal error happened while executing tests
# Exit code 4:  pytest command line usage error
# Exit code 5:  No tests were collected

tput bold
#echo -e "\nRunning $RUN_PREFIX libensemble Test-suite .......\n"
echo -e "\n************** Running: Libensemble Test-Suite **************\n"
tput sgr 0
echo -e "Selected:"
[ $RUN_UNIT_TESTS = "true" ] && echo -e "Unit Tests"
[ $RUN_COV_TESTS = "true" ]  && echo -e " - including coverage analysis"
[ $RUN_REG_TESTS = "true" ]  && echo -e "Regression Tests"
[ $RUN_PEP_TESTS = "true" ]  && echo -e "PEP Code Standard Tests (static code test)"

COV_LINE_SERIAL=''
COV_LINE_PARALLEL=''
if [ $RUN_COV_TESTS = "true" ]; then
   COV_LINE_SERIAL='--cov --cov-report html:cov_unit'
   #COV_LINE_PARALLEL='-m coverage run --parallel-mode --rcfile=../.coveragerc' #running in sub-dirs
   COV_LINE_PARALLEL='-m coverage run --parallel-mode' #running in regression dir itself
   
   #include branch coverage? eg. flags if never jumped a statement block... [see .coveragerc file]
   #COV_LINE_PARALLEL='-m coverage run --branch --parallel-mode'
fi;



if [ "$root_found" = true ]; then

  cd $ROOT_DIR/

  # Run Unit Tests -----------------------------------------------------------------------
  
  if [ "$RUN_UNIT_TESTS" = true ]; then  
    tput bold;tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running unit tests"
    tput sgr 0    
    
    cd $ROOT_DIR/$UNIT_TEST_SUBDIR  
    $PYTHON_RUN -m pytest $COV_LINE_SERIAL 
    code=$?
    if [ "$code" -eq "0" ]; then
      echo
      tput bold;tput setaf 2; echo "Unit tests passed. Continuing...";tput sgr 0
      echo
    else
      echo
      tput bold;tput setaf 1;echo -e "Abort $RUN_PREFIX: Unit tests failed: $code";tput sgr 0
      exit $code #return pytest exit code
    fi;
  fi;
  cd $ROOT_DIR/


  # Run Regression Tests -----------------------------------------------------------------
    
  if [ "$RUN_REG_TESTS" = true ]; then  
    tput bold;tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running regression tests"
    tput sgr 0
    
    cd $ROOT_DIR/$REG_TEST_SUBDIR
    
    #Check output dir exists.
    if [ ! -d output ]; then
      mkdir output/
    fi;

    #Running without subdirs - delete any leftover output and coverage data files
    cleanup
               
    #Build sim/gen dependencies
    cd $ROOT_DIR/$GKLS_BUILD_DIR 
    make gkls_single
    #make gkls
        
    #Add further dependencies here .....
            
    cd $ROOT_DIR/$REG_TEST_SUBDIR
        
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
    for TEST_SCRIPT in $REG_TEST_LIST
    do
      #Need proc count here for now - still stop on failure etc.
      for NPROCS in $REG_TEST_PROCESS_COUNT_LIST
      do      
        test_num=$((test_num+1))
        
        RUN_TEST=true
        if [ $REG_STOP_ON_FAILURE = "true" ]; then
          #Before Each Test check code is 0 (passed so far) - or skip to test summary        
          if [ "$code" -ne "0" ]; then
            RUN_TEST=false
            break
          fi
        fi

        if [ "$RUN_TEST" = "true" ]; then

           if [ "$REG_USE_PYTEST" = true ]; then
             mpiexec -np $NPROCS $PYTHON_RUN -m pytest $TEST_SCRIPT >> $TEST_SCRIPT.$NPROCS'procs'.$REG_TEST_OUTPUT_EXT 2>test.err
             #mpiexec -np $REG_TEST_PROCESS_COUNT $PYTHON_RUN -m pytest $COV_LINE_PARALLEL test_libE_on_GKLS_pytest.py >> $REG_TEST_OUTPUT
             test_code=$?
           else
             mpiexec -np $NPROCS $PYTHON_RUN $COV_LINE_PARALLEL $TEST_SCRIPT >> $TEST_SCRIPT.$NPROCS'procs'.$REG_TEST_OUTPUT_EXT 2>test.err
             test_code=$?
           fi
           reg_count_runs=$((reg_count_runs+1))

           if [ "$test_code" -eq "0" ]; then
             echo -e " ---Test $test_num: $TEST_SCRIPT on $NPROCS processes ${pass_color} ...passed ${textreset}"
             reg_pass=$((reg_pass+1))
             #continue testing
           else
             echo -e " ---Test $test_num: $TEST_SCRIPT on $NPROCS processes ${fail_color}  ...failed ${textreset}"
             code=$test_code #sh - currently stop on failure
             if [ $REG_STOP_ON_FAILURE != "true" ]; then
               #Dump error to log file
               echo -e "\nTest $test_num: $TEST_SCRIPT on $NPROCS processes:\n" >>log.err; cat test.err >>log.err
             fi;
             reg_fail=$((reg_fail+1))     
           fi;

           #If use sub-dirs - move this test's coverage files to regression dir where they can be merged with other tests
           #[ "$RUN_COV_TESTS" = "true" ] && mv .cov_reg_out.* ../

        fi; #if [ "$RUN_TEST" = "true" ];
      
      done #nprocs
      
      [ $REG_STOP_ON_FAILURE = "true" ] && [ "$code" -ne "0" ] && cat test.err && break
      reg_count_tests=$((reg_count_tests+1))
      
    done #tests
    reg_end=$(current_time)
    reg_time=$(total_time $reg_start $reg_end)
    
    # ********* End Loop over regression tests *********

    
    cd $ROOT_DIR/$REG_TEST_SUBDIR

    #Create Coverage Reports ----------------------------------------------

    #Only if passed
    if [ "$code" -eq "0" ]; then
      
      echo -e "\n..Moving output files to output dir"      
      if [ "$(ls -A output)" ]; then
        rm output/* #Avoid mixing test run results
      fi;
        
      #sh - shld active_runs be prefixed for each job
      filelist=(*.$REG_TEST_OUTPUT_EXT);   [ -e ${filelist[0]} ] && mv *.$REG_TEST_OUTPUT_EXT output/
      filelist=(*.npy);                    [ -e ${filelist[0]} ] && mv *.npy output/
      filelist=(*active_runs.txt);         [ -e ${filelist[0]} ] && mv *active_runs.txt output/
                    
      if [ "$RUN_COV_TESTS" = true ]; then
        
        # Merge MPI coverage data for all ranks from regression tests and create html report in sub-dir
        
        # Must combine all if in sep sub-dirs will copy to dir above
        coverage combine .cov_reg_out.* #Name of coverage data file must match that in .coveragerc in reg test dir.
        coverage html
        echo -e "..Coverage HTML written to dir $REG_TEST_SUBDIR/cov_reg/"
          
        if [ "$RUN_UNIT_TESTS" = true ]; then

          #Combine with unit test coverage at top-level
          cd $ROOT_DIR/$COV_MERGE_DIR
          cp $ROOT_DIR/$UNIT_TEST_SUBDIR/.cov_unit_out .
          cp $ROOT_DIR/$REG_TEST_SUBDIR/.cov_reg_out .
          
          #coverage combine --rcfile=.coverage_merge.rc .cov_unit_out .cov_reg_out
          coverage combine .cov_unit_out .cov_reg_out #Should create .cov_merge_out - see .coveragerc
          coverage html #Should create cov_merge/ dir
          echo -e "..Combined Unit Test/Regression Test Coverage HTML written to dir $COV_MERGE_DIR/cov_merge/"

        fi;
        
      fi;
    fi;


    #All reg tests - summary ----------------------------------------------
    if [ "$code" -eq "0" ]; then
      echo
      #tput bold;tput setaf 2
      
      if [ "$REG_USE_PYTEST" != true ]; then
        #sh - temp formatting similar(ish) to pytest - update in python (as with timing)
        #tput bold;tput setaf 4; echo -e "***Note***: temporary formatting/timing ......"
        
        summ_line="$reg_pass passed in $reg_time seconds"
        tput bold;tput setaf 2;
        print_summary_line $summ_line
        tput sgr 0
      fi;
      
      tput bold;tput setaf 2;echo -e "\nRegression tests passed ..."      
      tput sgr 0
      echo
    else
      if [ $REG_STOP_ON_FAILURE != "true" ]; then
        echo -e ""
        echo -e "\n..see error log at $REG_TEST_SUBDIR/log.err"
        summ_line="$reg_fail failed, $reg_pass passed in $reg_time seconds"
        tput bold;tput setaf 1;
        print_summary_line $summ_line
        tput sgr 0        
      fi;
      echo
      tput bold;tput setaf 1;echo -e "\nAbort $RUN_PREFIX: Regression tests failed (exit code $code)";tput sgr 0
      echo
      exit $code
    fi;  
    
  fi; #$RUN_REG_TESTS


  # Run Code standards Tests -----------------------------------------
  cd $ROOT_DIR
  if [ "$RUN_PEP_TESTS" = true ]; then
    tput bold;tput setaf 6
    echo -e "\n$RUN_PREFIX --$PYTHON_RUN: Running PEP tests - All python src below $PEP_SCOPE"
    tput sgr 0
    pytest --$PYTHON_PEP_STANDARD $ROOT_DIR/$PEP_SCOPE
    
    code=$?
    if [ "$code" -eq "0" ]; then
      echo
      tput bold;tput setaf 2; echo "PEP tests passed. Continuing...";tput sgr 0
      echo
    else
      echo
      tput bold;tput setaf 1;echo -e "Abort $RUN_PREFIX: PEP tests failed: $code";tput sgr 0
       exit $code #return pytest exit code
    fi;  
  fi;

  # ------------------------------------------------------------------ 
  tput bold;tput setaf 2; echo -e "\n$RUN_PREFIX --$PYTHON_RUN: All tests passed\n"; tput sgr 0
  exit 0
else
  tput bold;tput setaf 1; echo -e "Abort $RUN_PREFIX:  Git repository root not found"; tput sgr 0
  exit 1
fi
