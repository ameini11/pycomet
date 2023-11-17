// MIT License
//
// Copyright (c) 2020 Sergio Izquierdo
// Copyright (c) 2021 Florian
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*!
 *  @file       main.cpp
 *  @brief      Run EfficientNet on a cat image as an example.
 *  @details    Run an EfficientNet model on a cat image and print the result. 
 *              The EfficientNet model should be downloaded running create_model.py
 *  @author     Florian
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-09-16
 */

#include <tensorflow/c/c_api.h>

// CppFlow headers
#include <cppflow/cppflow.h>

// C++ headers
#include <iostream>


// bool DedicatedGPUAvailable(TF_Session* session){
//     if (session != nullptr) {
//         std::vector<tensorflow::DeviceAttributes> response;
//         session->ListDevices(&response);

//         // If a single device is found, we assume that it's the CPU. 
//         // You can check that name if you want to make sure that this is the case
//         if (response.size() == 1) {
//             std::cout << "No GPU found! Total available devices = 1" << std::endl;
//             std::cout << "Available devices: " << std::endl;
//             for (unsigned int i = 0; i < response.size(); ++i) {
//                 std::cout << i << " " << response[i].name() << std::endl;
//             }
//             return false;
//         }
//     }
//     return true;
// }

// void printDeviceDetails(const TF_DeviceList* deviceList, int deviceIndex, TF_Status* status) {

//     const char* deviceName = TF_DeviceListName(deviceList, deviceIndex, status);
//     const char* deviceType = TF_DeviceListType(deviceList, deviceIndex, status);
//     uint64_t memoryLimit = TF_DeviceListMemoryBytes(deviceList, deviceIndex, status);

//     printf("Device Name: %s\n", deviceName);
//     printf("Device Type: %s\n", deviceType);
//     printf("Memory Limit: %lu\n", memoryLimit);
//     printf("\n");
// }


// void checkDeviceAllocation(TF_Graph* graph) {
//     TF_Operation* operation = nullptr;
//     size_t pos = 0;

//     while ((operation = TF_GraphNextOperation(graph, &pos)) != nullptr) {
//         int numInputs = TF_OperationNumInputs(operation);
//         int numOutputs = TF_OperationNumOutputs(operation);

//         // Check input devices
//         for (int i = 0; i < numInputs; ++i) {
//             TF_Output input = {operation, i};
//             TF_Tensor* tensor = TF_OperationInput(input);

//             if (tensor != nullptr) {
//                 const char* deviceName = TF_DeviceName(TF_TensorDevice(tensor));
//                 printf("Input %d of operation %s is allocated to device %s\n", i, TF_OperationName(operation), deviceName);
//                 TF_DeleteTensor(tensor);
//             }
//         }

//         // Check output devices
//         for (int i = 0; i < numOutputs; ++i) {
//             TF_Output output = {operation, i};
//             TF_Tensor* tensor = TF_OperationOutput(output);

//             if (tensor != nullptr) {
//                 const char* deviceName = TF_DeviceName(TF_TensorDevice(tensor));
//                 printf("Output %d of operation %s is allocated to device %s\n", i, TF_OperationName(operation), deviceName);
//                 TF_DeleteTensor(tensor);
//             }
//         }
//     }
// }



int main() {

    int nb_iters = 1000;
    std::string MODEL_PATH = "model";
    std::string IMAGE_PATH = "image.jpg";

    auto input = cppflow::decode_jpeg(
        cppflow::read_file(IMAGE_PATH));

    std::cout << "input shape : " << input.shape() << std::endl;

    std::cout << "tensor::device(true) : " << input.device(true) << std::endl;
    std::cout << "tensor::device(false) : " << input.device(false) << std::endl;

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);

    std::cout << "tensor::device(true) : " << input.device(true) << std::endl;
    std::cout << "tensor::device(false) : " << input.device(false) << std::endl;

    input = cppflow::expand_dims(input, 0);
    cppflow::model model(MODEL_PATH);


    // TF_DeviceList* deviceList = TF_SessionListDevices(model.session.get(), model.status.get());

    // int numDevices = TF_DeviceListCount(deviceList);
    // for (int i = 0; i < numDevices; ++i) {
    //     printDeviceDetails(deviceList, i, model.status.get());
    // }

    // checkDeviceAllocation(model.graph.get());

    auto output = model(input);
    for (int i = 0; i < nb_iters; ++i) {
        output = model(input);
        std::cout << "Iter: " << i << std::endl;
    }

    // std::cout << "output shape : " << output.shape() << std::endl;
    // std::cout << "tensor::device(true) : " << output.device(true) << std::endl;
    // std::cout << "tensor::device(false) : " << output.device(false) << std::endl;

    // std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;

    return 0;
}
