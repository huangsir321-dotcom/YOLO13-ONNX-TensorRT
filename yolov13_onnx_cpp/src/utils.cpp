#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

Ort::Session create_session(Ort::Env& env, const char* model_path) {
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return Ort::Session(env, model_path, session_options);
}

std::vector<const char*> get_input_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_names(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        input_names[i] = session.GetInputNameAllocated(i, allocator).release();
    }
    
    return input_names;
}

std::vector<const char*> get_output_names(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_names(num_output_nodes);
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_names[i] = session.GetOutputNameAllocated(i, allocator).release();
    }
    
    return output_names;
}

std::vector<int64_t> get_input_shape(Ort::Session& session) {
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<int64_t> get_output_shape(Ort::Session& session) {
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}
