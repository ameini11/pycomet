# PyCOMET-API
**Physics-informed machine learning for scientific computing and operator discovery**

Run PyCOMET models in C++/Fortran without TensorFlow installation and compilation. You can take a look to the [examples](https://github.com/ivanZanardi/pycomet/tree/main/api/pycomet/examples) to see how to load a deep network in your C++/Fortran project.

PyCOMET-API uses [CppFlow](https://github.com/serizba/cppflow/) and [Tensorflow C API](https://www.tensorflow.org/install/lang_c) to run the models. With this project you can manage and run your models in C++/Fortran without worrying about void, malloc or free. With PyCOMET-API you easily can:

* Open saved models created with PyCOMET
* Execute Tensorflow neural networks in C++/Fortran
* Perform tensor manipulation directly from C++

### Dependencies

  - [Tensorflow C API](https://www.tensorflow.org/install/lang_c/)
  - [CppFlow](https://github.com/serizba/cppflow/)
  - [Json](https://github.com/nlohmann/json/)
