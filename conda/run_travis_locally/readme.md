# How to install Travis Python container locally and build libEnsemble

## Installing Travis Locally

This explains how to replicate the Travis container environment locally.

This can save a lot of git pushes and trial-and-error when there is problems in Travis builds that are not 
observed locally.

#### References:

[How to Run TravisCI locally on Docker](https://medium.com/google-developers/how-to-run-travisci-locally-on-docker-822fc6b2db2e)

[Travis CI Docker instructions](https://docs.travis-ci.com/user/common-build-problems/#troubleshooting-locally-in-a-docker-image)

I did it as follows on Ubuntu 18.04.

### Install Docker

Here are the lines for Ubuntu. Go one at a time. Taken from [here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04) (see for details or if issues with these lines).

    sudo apt update
    sudo apt install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
    sudo apt update
    apt-cache policy docker-ce
    sudo apt install docker-ce

Add user name to docker group so dont need sudo before docker:

    sudo usermod -aG docker ${USER}
    su - ${USER}
    id -nG # Confirm username added to docker users
    
    
### Install Travis Image for Python and create a container

Name your new travis container:

    export CONTAINER=travis-debug-3.7-v1

Install the Travis Docker image for Python and run the container (server):

    sudo docker run --name $CONTAINER -dit travisci/ci-garnet:packer-1512502276-986baf0 /sbin/init

This might take a while if image is not downloaded. Once image is downloaded it will be run from cache.

Note: 
travisci/ci-garnet:packer-1512502276-986baf0 is the Travis Python image. The name in
$CONTAINER is the name you are assigning to the new container made from the image.

Alternative travisCI docker images can be found [here](https://hub.docker.com/r/travisci/ci-garnet/tags/).


### Start a shell session in the container

Then open a shell in running container:

    sudo docker exec -it $CONTAINER bash -l
    
Prompt should say travis@ rather than root@:


### Build and run libEnsemble for a given Python version

You now want to copy the latest build script to the container and run it.

The default build script is build_mpich_libE.sh. If this is not up to date, check
the installs against .travis.yml in the top level libEnsemble package directory.


#### Copy build script from host system to the running container

Run the following **from your host machine environment**. This copies the given file into an existing 
container named $CONTAINER. Where /home/travis is target location in container
filesystem.

    docker cp build_mpich_libE.sh $CONTAINER:/home/travis

On the docker side you may need to set ownership of the script:

    chown travis:travis /home/travis/build_mpich_libE.sh

    
#### Run the libEnsemble build-and-run script

Now, in the docker container, become user travis:

    su - travis
 
Source the script, setting python version and libEnsemble branch if necessary (see script for defaults).
**Always source** to maintain environment variables after running (inc. miniconda path). The following
would run with Python 3.7 and test the libEnsemble branch hotfix/logging:

    . ./build_mpich_libE.sh -p 3.7 -b hotfix/logging
    
Note: libEnsemble will be git cloned and checked out at the given branch.

The script should build all dependencies and run tests. Having sourced the script, you should
be left in the same environment for debugging. The script should stop running if an install step fails.

If the conda output is too verbose, remove the "set -x" line in the script.


### Saving images from built containers and re-starting containers

To exit the container session (client).
    exit # Travis user
    exit # Exit running session (but container server is still running.
    
Re-enter while background container still running:

    sudo docker exec -it $CONTAINER bash -l
    su - travis
    
Note that environment variables are not saved.

To save the modified container as an image (with your changes to filesystem):

Note: This will not by default save your environment variables

First to check running containers (ie. running server - whether or not you are in a session):

    docker ps
    
OR

    docker container ls

The first column of output is the container ID.

To save an image locally from this container:

    docker commit -m "Optional log message" <container ID> <user_name>/<image_name>

Where <user_name> is your user name on the machine and <image_name> is what you
want to call your new image.

Now it should show if you do:

    docker images

If it is saved you can stop the container (server) thats running and restart eg.

    docker stop $CONTAINER
    
where $CONTAINER is the name you gave the container on the docker run command.

You can restart from your new image with docker run and docker exec or to run server and run session in one:

    docker run -it <user_name>/<image_name> /bin/bash
    
where <user_name>/<image_name> is as used above to save (or from first column in "docker images" output).


### Notes

The CPU cores available will be all those on your machine, not just what Travis supports.
Parallel/threaded code run-time errors may well depend on timing of processes/threads and
so may not be replicated. At time of writing, Travis provides 2 cores. Resources in docker
container can be [restricted](https://docs.docker.com/config/containers/resource_constraints/).

An alternative to this process is to log in to your Travis build and debug. For public
repositories, this requires contacting Travis support for a token that can only be used by
the given user. See [here](https://docs.travis-ci.com/user/running-build-in-debug-mode/).


    
