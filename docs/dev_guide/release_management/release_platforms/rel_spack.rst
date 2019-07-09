.. _rel-spack:

A workflow for updating libEnsemble on Spack
============================================

Introduction
------------

This assumes you have already:

 - Made a PyPI package for new version of libensemble
 - Made a github fork of Spack and cloned it to your local system.

Details on how to create forks can be found at: https://help.github.com/articles/fork-a-repo

You now have a configuration like shown in answer at: https://stackoverflow.com/questions/6286571/are-git-forks-actually-git-clones

Upstream, in this case, is the official Spack repository on github. Origin is your fork on github and Local Machine is your local clone (from your fork).


Make sure SPACK_ROOT is set and spack binary is in your path::

    export SPACK_ROOT=<PATH/TO/LOCAL/SPACK/REPO>
    export PATH=$SPACK_ROOT/bin:$PATH


DO ONCE in your local checkout:

To set upstream repo::

    git remote add upstream https://github.com/spack/spack.git
    git remote -v # check added

(Optional) Prevent accidental pushes to upstream::

    git remote set-url --push upstream no_push
    git remote -v # Check for line: `upstream	no_push (push)`
    

Now to update (the main develop branch)
---------------------------------------

You will now update your local machine from the upstream repo (if in doubt - make a copy of local repo
in your filestystem before doing the following).

Check upstream remote is present::

    git remote -v

Ensure you are on the develop branch::

    git checkout develop

Fetch from upstream repo::

    git fetch upstream

Now to update your local machine you may wish to rebase or overwrite your local files.
Select from the following:

If you have local changes to go "on top" of latest code::

    git rebase upstream/develop

Or to make your local machine identical to upstream repo (**WARNING** Any local changes WILL BE LOST)::

    git reset --hard upstream/develop

    
(Optional) You may want to update your forked (origin) repo on github at this point.
This may requires a forced push::

    git push origin develop --force
    

Making changes
--------------

You can optionally create a branch to make changes on. This may be a good idea, especially if
you have multiple packages, to make separate branches for each package.

See the Spack [packaging](https://spack.readthedocs.io/en/latest/packaging_guide.html) and
[contibution](https://spack.readthedocs.io/en/latest/contribution_guide.html) guides for more info.


Quick example to update libensemble::

    git branch update_libensemble
    git checkout update_libensemble
    
This will open the libensemble package.py file in your editor (given by env variable EDITOR)::

    spack edit py-libensemble  # SPACK_ROOT must be set (see above) (python packages use "py-" prefix)

Or just open it manually: var/spack/repos/builtin/packages/py-libensemble/package.py
    
    
Now get checksum for new lines:

Get the tarball (see PyPI instructions), for the new release and use::

    sha256sum libensemble-*.tar.gz

Update the `package.py` file by pasting in the new checksum lines (and make sure url line points to latest version).
Also update any dependencies for the new version.

Check package::

     spack flake8

If OK add, commit and push to origin (forked repo)::

     git commit -am "Update libensemble"
     git push origin update_libensemble --force
     
Once the branch is pushed to the forked repo - go to github and do a pull request from this
branch on the fork to the develop branch on the upstream.

    
Express summary: Make fork identical to upstream
------------------------------------------------

Quick summary for bringing develop branch on forked repo up to speed with upstream
(YOU WILL LOSE ANY CHANGES)::

    git remote add upstream https://github.com/spack/spack.git  
    git fetch upstream  
    git checkout develop  
    git reset --hard upstream/develop  
    git push origin develop --force  

Reference: <https://stackoverflow.com/questions/9646167/clean-up-a-fork-and-restart-it-from-the-upstream/39628366>
    
