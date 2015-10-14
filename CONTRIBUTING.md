# Contributing


## Releasing a new version

The release process includes several stages, most of which are automated:

- declaring a new version
- building and deploying the package to a pacakge index
- building and deploying the documentation

A more elaborate scheme can be found [here](https://khmer.readthedocs.org/en/v1.1/release.html).

### Declaring a new version

This is done by tagging a commit with a new version number which must conform to PEP440 (v#.#.#).
No change in the code itself is required as everything is handled by [versioneer](https://github.com/warner/python-versioneer).

When all tests pass and everything is commited:

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