// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include <functional>

#include "lib/Api.Ort/pch.h"

#include "AdapterSessionTest.h"
#include "ILotusValueProviderPrivate.h"
#include "onnxruntime_c_api.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"
#include "OnnxruntimeModel.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_env.h"

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

namespace {
WinML::OnnxruntimeEngineFactory *engine_factory;
const OrtApi *ort_api;
const WinmlAdapterApi *winml_adapter_api;
OrtEnv* ort_env;

void AdapterSessionTestSetup() {
  init_apartment();
  WINML_EXPECT_HRESULT_SUCCEEDED(Microsoft::WRL::MakeAndInitialize<WinML::OnnxruntimeEngineFactory>(&engine_factory));
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory->GetOrtEnvironment(&ort_env));
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory->EnableDebugOutput(true));
  WINML_EXPECT_NOT_EQUAL(nullptr, winml_adapter_api = engine_factory->UseWinmlAdapterApi());
  WINML_EXPECT_NOT_EQUAL(nullptr, ort_api = engine_factory->UseOrtApi());
}

UniqueOrtSessionOptions CreateUniqueOrtSessionOptions() {
  OrtSessionOptions *options;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&options), ort_api);
  return UniqueOrtSessionOptions(options, ort_api->ReleaseSessionOptions);
}

void AppendExecutionProvider_CPU() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), true), ort_api);
}

ID3D12Device* CreateD3DDevice() {
  ID3D12Device* device = nullptr;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device)));
  return device;
  // TODO Check for leaks
}

ID3D12CommandQueue* CreateD3DQueue(ID3D12Device* device) {
  ID3D12CommandQueue* queue = nullptr;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  device->CreateCommandQueue(&command_queue_desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue));
  return queue;
  // TODO Check for leaks
}

UniqueOrtSession CreateUniqueOrtSession(const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateUniqueOrtSession(const std::wstring& model_path, const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  ort_api->SetIntraOpNumThreads(session_options.get(), 1);
  ort_api->SetSessionGraphOptimizationLevel(session_options.get(), ORT_ENABLE_BASIC);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), true), ort_api);
  // ort_api->DisableMemPattern(session_options.get()), ort_api);
  // const auto device = CreateD3DDevice();
  // const auto queue = CreateD3DQueue(device);
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device, queue), ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->CreateSession(ort_env, model_path.c_str(), session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

void AppendExecutionProvider_DML() {
  GPUTEST
  const auto session_options = CreateUniqueOrtSessionOptions();

  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device, queue), ort_api);
}

void CreateWithoutModel() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  CreateUniqueOrtSession(session_options);
}

void GetExecutionProvider_CPU() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  const auto model_path = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  auto session = CreateUniqueOrtSession(model_path, session_options);

  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
}

void GetExecutionProvider_DML() {
  GPUTEST
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);
  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device, queue), ort_api);

  const auto model_path = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  auto session = CreateUniqueOrtSession(model_path, session_options);

  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  // Test if DML EP method can be called
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider), ort_api);
}

void RegisterGraphTransformers() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  winml_adapter_api->SessionRegisterGraphTransformers(session.get());
}

void RegisterGraphTransformers_DML() {
  GPUTEST
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  winml_adapter_api->SessionRegisterGraphTransformers(session.get());
}

void RegisterCustomRegistry() {
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  if (registry) {
    const auto session_options = CreateUniqueOrtSessionOptions();
    auto session = CreateUniqueOrtSession(session_options);
    THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionRegisterCustomRegistry(session.get(), registry), ort_api);
  }
}

void RegisterCustomRegistry_DML() {
  GPUTEST
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, registry);
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionRegisterCustomRegistry(session.get(), registry), ort_api);
}

void LoadAndPurloinModel(const UniqueOrtSession& session, const std::string& model_path) {
  com_ptr<WinML::IModel> model;
  WINML_THROW_IF_FAILED(engine_factory->CreateModel(model_path.c_str(), sizeof(model_path), model.put()));

  Microsoft::WRL::ComPtr<WinML::IOnnxruntimeModel> onnxruntime_model;
  WINML_EXPECT_HRESULT_SUCCEEDED(model->QueryInterface(IID_PPV_ARGS(&onnxruntime_model)));
  OrtModel* ort_model;
  WINML_EXPECT_HRESULT_SUCCEEDED(onnxruntime_model->DetachOrtModel(&ort_model));
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);
}

void LoadAndPurloinModel() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  LoadAndPurloinModel(session, "fns-candy.onnx");
}

void Initialize() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);

  com_ptr<WinML::IModel> model;
  const auto model_path = "fns-candy.onnx";
  WINML_THROW_IF_FAILED(engine_factory->CreateModel(model_path, sizeof(model_path), model.put()));

  Microsoft::WRL::ComPtr<WinML::IOnnxruntimeModel> onnxruntime_model;
  WINML_EXPECT_HRESULT_SUCCEEDED(model->QueryInterface(IID_PPV_ARGS(&onnxruntime_model)));
  OrtModel* ort_model;
  WINML_EXPECT_HRESULT_SUCCEEDED(onnxruntime_model->DetachOrtModel(&ort_model));
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
}

class MockSink : public onnxruntime::logging::ISink {
public:
  static bool called;

private:
  void SendImpl(const onnxruntime::logging::Timestamp& timestamp, const std::string& logger_id, const onnxruntime::logging::Capture& message) override {
    called = true;
  }

  void SendProfileEvent(onnxruntime::profiling::EventRecord&) const override { };
};
bool MockSink::called = false;

void Profiling() {
  ort_env->SetLoggingManager(nullptr);
  const std::string logger_name("Default");
  auto logging_manager = std::make_unique<onnxruntime::logging::LoggingManager>(
    std::make_unique<MockSink>(),
    onnxruntime::logging::Severity::kINFO,
    false,
    onnxruntime::logging::LoggingManager::InstanceType::Default,
    &logger_name);
  ort_env->SetLoggingManager(std::move(logging_manager));

  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);

  winml_adapter_api->SessionStartProfiling(ort_env, session.get());
  LoadAndPurloinModel(session, "fns-candy.onnx");
  winml_adapter_api->SessionEndProfiling(session.get());
  WINML_EXPECT_TRUE(MockSink::called);
}

void CopyInputAcrossDevices() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(L"fns-candy.onnx", session_options);

  constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  constexpr size_t input_tensor_size = [&dimensions]() {
    size_t size = 1;
    for (auto dim : dimensions)
      size *= dim;
    return size;
  } ();

  OrtMemoryInfo* memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), ort_api);
  std::vector<float> input_tensor_values(input_tensor_size);
  OrtValue* input_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort_api);

  int is_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->IsTensor(input_tensor, &is_tensor), ort_api);
  WINML_EXPECT_TRUE(is_tensor);

  OrtValue* dest_ort_value = nullptr;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionCopyOneInputAcrossDevices(session.get(), "inputImage", input_tensor, &dest_ort_value), ort_api);

  ort_api->ReleaseMemoryInfo(memory_info);
}

void CopyInputAcrossDevices_DML() {
  GPUTEST

  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);
  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device, queue), ort_api);
  auto session = CreateUniqueOrtSession(session_options);

  LoadAndPurloinModel(session, "fns-candy.onnx");
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
  constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  constexpr size_t input_tensor_size = [&dimensions]() {
    size_t size = 1;
    for (auto dim : dimensions)
      size *= dim;
    return size;
  } ();

  OrtMemoryInfo* memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), ort_api);
  std::vector<float> input_tensor_values(input_tensor_size);
  OrtValue* input_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort_api);

  int is_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->IsTensor(input_tensor, &is_tensor), ort_api);
  WINML_EXPECT_TRUE(is_tensor);

  OrtValue* dest_ort_value = nullptr;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionCopyOneInputAcrossDevices(session.get(), "inputImage", input_tensor, &dest_ort_value), ort_api);

  ort_api->ReleaseMemoryInfo(memory_info);
}
}

const AdapterSessionTestAPi& getapi() {
  static constexpr AdapterSessionTestAPi api =
  {
    AdapterSessionTestSetup,
    AppendExecutionProvider_CPU,
    AppendExecutionProvider_DML,
    CreateWithoutModel,
    GetExecutionProvider_CPU,
    GetExecutionProvider_DML,
    Initialize,
    RegisterGraphTransformers,
    RegisterGraphTransformers_DML,
    RegisterCustomRegistry,
    RegisterCustomRegistry_DML,
    LoadAndPurloinModel,
    Profiling,
    CopyInputAcrossDevices,
    CopyInputAcrossDevices_DML
  };
  return api;
}
