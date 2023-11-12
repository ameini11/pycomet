# PyCOMET-API
**Physics-informed machine learning for scientific computing and operator discovery**


PyCOMET-API enables the integration of PyCOMET models into C++/Fortran projects seamlessly, without the need for TensorFlow installation or compilation. Explore the [examples](https://github.com/ivanZanardi/pycomet/tree/main/api/pycomet/examples) to understand how to incorporate a TensorFlow `tf.Model` into your C++/Fortran project.

PyCOMET-API utilizes [CppFlow](https://github.com/serizba/cppflow/) and the [TensorFlow C API](https://www.tensorflow.org/install/lang_c) to execute models. This project simplifies the management and execution of models in C++/Fortran, handling concerns about void, malloc, or free operations. With PyCOMET-API, you can effortlessly:
* Open saved models created with PyCOMET
* Execute TensorFlow neural networks in C++/Fortran
* Perform tensor manipulation directly from C++

### Dependencies

* [TensorFlow C API](https://www.tensorflow.org/install/lang_c/) = 2.10.0
* [CppFlow](https://github.com/serizba/cppflow/) = 2.0.0
* [Json](https://github.com/nlohmann/json/)

### Installation

To install PyCOMET-API, follow these steps after downloading the dependencies:
1. Define installation directories
```
export SRC_DIR=<path-to-PyCOMET-API-source-dir>
export BUILD_DIR=<path-to-build-dir>
export INSTALL_DIR=<path-to-install-dir>
```
2. Define dependencies directories
```
export TensorFlow_DIR=<path-to-tensorflow-dir>
export CppFlow_DIR=<path-to-cppflow-dir>
export Json_DIR=<path-to-json-dir>
```
3. Prepare installation directories
```
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
```
4. Configure CMake
```
cmake $SRC_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_EXAMPLES=OFF \
  -Djson_INCLUDE_DIR=$Json_DIR/include \
  -Dcppflow_INCLUDE_DIR=$CppFlow_DIR/include \
  -Dtensorflow_INCLUDE_DIR=$TensorFlow_DIR/include \
  -Dtensorflow_LIBRARY=$TensorFlow_DIR/lib/libtensorflow.so
```
5. Install
```
make -j 4
make install
```

### GPU Support

The code has been tested with the following configuration:

| Compiler   | cuDNN | CUDA | TensorFlow C API |
|------------|-------|------|------------------|
| GCC 9.3.0  | 8.1   | 11.2 | 2.10.0           |
