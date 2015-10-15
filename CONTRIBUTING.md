# Contributing

## Releasing a new version

The release process includes several stages, most of which are automated:

- review changes
- test locally
- declaring a new version
- building and deploying the package to a pacakge index
- building and deploying the documentation

A more elaborate scheme can be found [here](https://khmer.readthedocs.org/en/v1.1/release.html).

### Review changes

Check that everything is commited:

```
git status
```

Review difference from the origin master branch:

```
git diff origin/master
```

and the commit log:

```
git log
```

Consider rebasing commits (only those that were not pushed! master branch is protected from force push):

```
git rebase -i HEAD~#
```

If all changes are commited and you are sure you are interested in all the commits, you can continue.

### Run tests locally

If you don't have a local copy of the code, you have to install curveball with the `[tests]` modifier:

```
pip install curveball[tests]
```

To run the tests:

```
nosetests -v tests
```

### Test build

Try to build the package.

```
python setup.py sdist
```

will create a new `curveball-<version>.zip` or similar package file in a folder called `dist`.

Now create a new environment using `conda` (or `virtualenv`, but that might cause some problems when installing dependencies)
and try to install Curveball from the distribution we built:

```
conda create -n test python
activate test
```

Follow the directions in the documentation on how to install the dependencies (skipping this might work, but not guaranteed), 
then install Curveball from the distribution you just built:

```
pip install dist/curveball-<version>.zip
```

Verify the installation:

```
curveball --version
```

If everything worked out you can remove the virtualenv:
```
deactivate
conda env remove -n test
```

### Declaring a new version

This is done by tagging a commit with a new version number which must conform to PEP440 (v#.#.#).
No change in the code itself is required as everything is handled by [versioneer](https://github.com/warner/python-versioneer).

```
git tag v#.#.#
git push
git push --tags
```

### Building and deplyoing the package

Pushing the tags will cause [Travis-CI](https://magnum.travis-ci.com/yoavram/curveball)
to pull the code, install dependencies, and test the code. 
If the tests succeed, Travis-CI will build the package and deploy it to PyPi,
but only on tagged commits from the `master` branch.


### Building and deplyoing the documentation

Travis-CI will also build the documentation and deploy it (currently to [divshot.io](https://curveball.divshot.io)).
