if [[ -z $1 ]]
then
  echo -e "Need output dir name: ./clone NEW_DIR"
else
  NEWDIR=$1

  INPUTS="sim/ forces.c forces_simf.py \
          run_libe_forces.py forces.x \
          *.sh"
  # Take you there if source script.
  mkdir ../$NEWDIR && cp -rp $INPUTS ../$NEWDIR && cd ../$NEWDIR
fi
