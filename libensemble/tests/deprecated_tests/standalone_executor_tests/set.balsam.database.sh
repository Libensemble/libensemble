# Source this once in env you will run in (e.g. On MOM node on Theta)

#Where you want your Balam database to be
export BALSAM_DB_PATH=~/database_sqlite

#Create a new database is not setup.
if [[ ! -d $BALSAM_DB_PATH ]]; then
# if [[ ! -f $BALSAM_DB_PATH/dbwriter_address ]]; #Ensures a database - may only be sql though
  echo -e '\nCreating database at $BALSAM_DB_PATH'
  balsam init $BALSAM_DB_PATH --db-type sqlite3
fi;

#Ensure no existing database server running - and refresh
balsam dbserver --stop
balsam dbserver --reset $BALSAM_DB_PATH
balsam dbserver --start

echo -e '\nHint: Use <balsam which> to get info on current database'
