# blazingdb-communication
blazingdb communication framework

# Dependencies
- General dependencies: https://github.com/BlazingDB/blazingdb-toolchain
- gdrcopy v1.3
- ucx v1.5.0

## Dependencies building 
`apt-get update && apt-get install -y kmod git dh-autoreconf valgrind libnuma-dev`

gdropy:
```
git clone https://github.com/nvidia/gdrcopy
git checkout v1.3
make PREFIX=$PWD/prefix92 CUDA=/usr/local/cuda/ all install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/tmp/gdrcopy/prefix92/lib64/
./validate
./copybw
```

Ucx:
```
git clone https://github.com/openucx/ucx
git checkout v1.5.0
./autogen.sh
./contrib/configure-devel --prefix=$PWD/prefix92/ --with-cuda=/usr/local/cuda/ --with-gdrcopy=/tmp/gdrcopy/prefix92/ --without-java
make -j8
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/tmp/ucx/prefix92/lib/
./prefix92/bin/ucx_info -d
```

## with docker
`nvidia-docker run --rm --privileged --cap-add=ALL -ti -v /home/nfs/bd.wmalpica/ucx_demo/:/tmp/ -v /lib/modules:/lib/modules -v /usr/src:/usr/src nvidia/cuda:9.2-devel-ubuntu16.04 bash`

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
