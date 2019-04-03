# blazingdb-communication
blazingdb communication framework

## deps

```
# apt-get install -y --no-install-recommends libcurl4-openssl-dev libssl-dev
```

## Build

```
cmake -DBLAZINGDB_DEPENDENCIES_INSTALL_DIR=/foo/blazing-dependencies/ -DCMAKE_INSTALL_PREFIX:PATH=/bar/install ..
```

## Run tests

```
$ make blazingdb-communication-test
$ ./src/blazingdb/communication/blazingdb-communication-test
```

## Get coverage

```
$ make coverage
```

Open _coverage-html/index.html_
