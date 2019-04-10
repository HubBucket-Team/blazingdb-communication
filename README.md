# blazingdb-communication
blazingdb communication framework

# Dependencies
- General dependencies: https://github.com/BlazingDB/blazingdb-toolchain

## Build

```
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBLAZINGDB_DEPENDENCIES_INSTALL_DIR=/foo/blazingsql/dependencies/ \
      -DCMAKE_INSTALL_PREFIX:PATH=/foo/blazingdb_communication_install_dir/ \
      ..
```

**NOTE:**
If you want to build the dependencies using the old C++ ABI, add this cmake argument:

```bash
-DCXX_OLD_ABI=ON
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
