# PyCOMET-API
**Physics-informed machine learning for scientific computing and operator discovery**

Run PyCOMET models in C++/Fortran without TensorFlow installation and compilation. You can take a look to the [examples](https://github.com/ivanZanardi/pycomet/tree/main/api/pycomet/examples) to see how to load a TensorFlow `tf.Model` in your C++/Fortran project.

PyCOMET-API uses [CppFlow](https://github.com/serizba/cppflow/) and [TensorFlow C API](https://www.tensorflow.org/install/lang_c) to run the models. With this project you can manage and run your models in C++/Fortran without worrying about void, malloc or free. With PyCOMET-API you easily can:

* Open saved models created with PyCOMET
* Execute TensorFlow neural networks in C++/Fortran
* Perform tensor manipulation directly from C++

### Dependencies

  - [TensorFlow C API](https://www.tensorflow.org/install/lang_c/) = 2.9.1
  - [CppFlow](https://github.com/serizba/cppflow/)
  - [Json](https://github.com/nlohmann/json/)

### Installation

To install PyCOMET-API, you just have to download the abovementioned dependencies and then follow these steps:
- Define installation directories
```
export SRC_DIR=<path-to-PyCOMET-API-source-dir>
export BUILD_DIR=<path-to-build-dir>
export INSTALL_DIR=<path-to-install-dir>
```
- Define dependencies directories
```
export TensorFlow_DIR=<path-to-tensorflow-dir>
export CppFlow_DIR=<path-to-cppflow-dir>
export Json_DIR=<path-to-json-dir>
```
- Prepare installation directories
```
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
```
- Configure CMake
```
cmake $SRC_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_EXAMPLES=OFF \
  -Djson_INCLUDE_DIR=$Json_DIR/include \
  -Dcppflow_INCLUDE_DIR=$CppFlow_DIR/include \
  -Dtensorflow_INCLUDE_DIR=$TensorFlow_DIR/include \
  -Dtensorflow_LIBRARY=$TensorFlow_DIR/lib/libtensorflow.so
```
- Install
```
make -j 4
make install
```
