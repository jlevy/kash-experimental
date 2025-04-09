# kash-experimental

See the main [kash](https://github.com/jlevy/kash) repo for general instructions.

To run kash with the the experimental kit features enabled, ensure you have uv set up
then:

```shell
uv tool install kash-experimental --upgrade --force
kash
```

Or for dev builds from within this git repo:

```shell
# Install all deps and run tests:
make
# Run kash with all experimental kit features enabled:
uv run kash
```

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
