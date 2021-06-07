# Quick run

If you have followed the ``readme`` to obtain ``docker`` and set up a container, this gives a quick example of running it.

Note: In this examnple the tests are not automatically run (-i in option to build_mpich_libE.sh).

For details see the ``readme``.

Two windows:
Window 1: In install/run_travis_locally direcory.
Window 2: Will create and run container.

Windows 1 and 2 - name container. E.g:

    export CONTAINER=travis-debug-2020-07-20-py3.6

Window 2:

    sudo docker run --name $CONTAINER -dit travisci/ci-garnet:packer-1512502276-986baf0 /sbin/init
    sudo docker exec -it $CONTAINER bash -l

Window 1:

    docker cp build_mpich_libE.sh $CONTAINER:/home/travis

Window 1 Optional - user scripts to help navigate:

    docker cp ~/.bashrc  $CONTAINER:/home/travis
    docker cp ~/.alias   $CONTAINER:/home/travis

WWindow 2 (Example: Do not run tests python 3.6 - git branch feature/register_apps):

    chown travis:travis /home/travis/build_mpich_libE.sh
    su - travis
    . ./build_mpich_libE.sh -p 3.6 -b feature/register_apps -i

Window 2 Optional - user scripts to help navigate:

    . ~/.bashrc

Note: libEnsemble will be git cloned and checked out at the given branch.
