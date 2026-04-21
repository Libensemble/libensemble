if [[ "$OSTYPE" == "darwin"* ]]; then
    mpicc -cc=clang -g -o burn_time.x burn_time.c
    mpicc -cc=clang -g -o sleep_and_print.x sleep_and_print.c
else
    mpicc -g -o burn_time.x burn_time.c
    mpicc -g -o sleep_and_print.x sleep_and_print.c
fi
