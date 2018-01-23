#!/usr/bin/env bash

# This should work for listed packages. Files/modules can be added/changed, but if add new
# packages will need to add to this.

#Set to exit script on error
set -e


if [[ -z $1 ]]
then
  echo -e "Aborting: No repo supplied"
  echo -e "Usage E.g: ./convert-to-importable.sh libensemble-balsam/"
  echo -e "If in project root dir do ./convert-to-importable.sh ."
  exit
fi

echo -e "\nTime of writing: check sim_dir_name directory in test_branin_aposmm.py"
echo -e "Also remember need packages in setup.py and make sure got __init__.py in test dirs unit/regression"
echo -e "then can do <pip install .> or <pip install --upgrade .> from the project root dir. And try run tests.\n" 

REPO_DIR=${PWD}/$1
REPO_DIR=${REPO_DIR%/} #Remove if trailing slash

export CODE_DIR=$REPO_DIR/code

#List python package dirs
export LIBE_DIR=$CODE_DIR/src
export SIM_FUNCS_DIR=$CODE_DIR/examples/sim_funcs
export BRANIN_DIR=$SIM_FUNCS_DIR/branin
export GEN_FUNCS_DIR=$CODE_DIR/examples/gen_funcs
export ALLOC_FUNCS_DIR=$CODE_DIR/examples/alloc_funcs
export UTESTS_DIR=$CODE_DIR/tests/unit_tests
export REG_TESTS_DIR=$CODE_DIR/tests/regression_tests


# ----------------------------------------------------------------------

#Replace imports lines for each listed package - and remove sys.path.append lines
convert_import_paths() {
  echo ${PWD}
  
  sed -i -e "s/^\(\s*\)sys.path.append/\1#sys.path.append/g" *.py 
  
  for file in $LIBE_FILES
  do 
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.$filebase as $filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.$filebase/g" *.py
  done
  
  for file in $SIM_FUNCS_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.sim_funcs.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.sim_funcs.$filebase as $filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.sim_funcs.$filebase/g" *.py    
  done

  for file in $BRANIN_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.sim_funcs.branin.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.sim_funcs.branin.$filebase as $filebase/g" *.py    
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.sim_funcs.branin.$filebase/g" *.py    
  done

  for file in $GEN_FUNCS_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.gen_funcs.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.gen_funcs.$filebase as $filebase/g" *.py    
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.gen_funcs.$filebase/g" *.py    
  done
  
  for file in $ALLOC_FUNCS_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.alloc_funcs.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.alloc_funcs.$filebase as $filebase/g" *.py    
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.alloc_funcs.$filebase/g" *.py    
  done

  for file in $UTESTS_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.unit_tests.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.unit_tests.$filebase as $filebase/g" *.py    
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.unit_tests.$filebase/g" *.py    
  done

  for file in $REG_TESTS_FILES
  do
    filebase=${file%.py}
    sed -i -e "s/^\(\s*\)from $filebase/\1from libensemble.regression_tests.$filebase/g" *.py
    sed -i -e "s/^\(\s*\)import $filebase\s*$/\1import libensemble.regression_tests.$filebase as $filebase/g" *.py    
    sed -i -e "s/^\(\s*\)import $filebase/\1import libensemble.regression_tests.$filebase/g" *.py    
  done
  
  #Kludge to deal with using data file known_minima_and_func_values
  line_in="sim_dir_name='..\/..\/examples\/sim_funcs\/branin'"
  line_out="import pkg_resources; sim_dir_name=pkg_resources.resource_filename('libensemble.sim_funcs.branin', '.')"
  sed -i -e "s/${line_in}/${line_out}/g" *.py
  
  line_in="minima_and_func_val_file = os.path.join(sim_dir_name, 'known_minima_and_func_values')"
  line_out="import pkg_resources; minima_and_func_val_file = pkg_resources.resource_filename('libensemble.sim_funcs.branin', 'known_minima_and_func_values')"
  sed -i -e "s/${line_in}/${line_out}/g" *.py
   
}

# ----------------------------------------------------------------------
#List of files in each dir
LIBE_FILES=`find $LIBE_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
SIM_FUNCS_FILES=`find $SIM_FUNCS_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
BRANIN_FILES=`find $BRANIN_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
GEN_FUNCS_FILES=`find $GEN_FUNCS_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
ALLOC_FUNCS_FILES=`find $ALLOC_FUNCS_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
UTESTS_FILES=`find $UTESTS_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`
REG_TESTS_FILES=`find $REG_TESTS_DIR -maxdepth 1 -name "*.py" -exec basename {} \;`

echo -e "Converting libensemble src dir:"
cd $REPO_DIR

cd code/src
convert_import_paths

echo -e "Converting libensemble examples:"
cd ../examples/

cd calling_scripts
convert_import_paths

cd ../alloc_funcs
convert_import_paths

cd ../sim_funcs
convert_import_paths
cd branin/
convert_import_paths
cd ..

cd ../gen_funcs
convert_import_paths
cd ..

echo -e "Converting libensemble tests:"
cd ../tests/
cd unit_tests
convert_import_paths

cd ../regression_tests
convert_import_paths
