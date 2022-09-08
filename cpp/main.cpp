#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

// Load
SavedModelBundleLite model_bundle;  // Light portable version of the model
SessionOptions session_options = SessionOptions();
RunOptions run_options = RunOptions();
Status status = LoadSavedModel(     // Model decoding
    session_options,
    run_options,
    export_dir,
    {kSavedModelTagServe},
    &model_bundle
);

// Inference
outputLayer = {                     // Multi-outputs: identifiers definition
    "StatefulPartitionedCall:0",
    "StatefulPartitionedCall:4",
    "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:5",
    "StatefulPartitionedCall:3"
};
Status runStatus = model_bundle.GetSession()->Run(          // Inference
    {{"serving_default_input_tensor:0", input_tensor}},
    outputLayer,
    {},
    &outputs
);
