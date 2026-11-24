#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>

Ort::Session create_session(Ort::Env& env, const char* model_path);
std::vector<const char*> get_input_names(Ort::Session& session);
std::vector<const char*> get_output_names(Ort::Session& session);
std::vector<int64_t> get_input_shape(Ort::Session& session);
std::vector<int64_t> get_output_shape(Ort::Session& session);
