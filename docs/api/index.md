---
title: API Docs
nav_order: 5
---
# ORT API docs
{: .no_toc }

At the heart of the ORT API is the `inference session`. Once the session is created (with an ONNX model, and the options appropriate to your scenario), you call the session's `run` supplying inputs and collecting outputs represented as `ORT values`.

An ORT value is a flexible and performant data structure that is mapped to the native data types in each language API, as well as being able to be bound to different hardware devices to minimize the amount of copying between devices using the `IO binding` API.

To use IO binding, you "bind" each input and output to a device of choice. By default all inputs and outputs are bound to the CPU, so if that is your scenario, you do not have to call the IO binding API.

You can see more details of how the API works in each of the language API reference sections below.


|:----------------------------------------------------------------------------------|
| <span class="fs-5"> [Python API Docs](https://onnxruntime.ai/docs/api/python/api_summary.html){: .btn target="_blank"} </span>  | 
| <span class="fs-5"> [Java API Docs](https://onnxruntime.ai/docs/api/java/index.html){: .btn target="_blank"} </span>   | 
| <span class="fs-5"> [C# API Docs](./csharp-api){: .btn target="_blank"} </span>|
| <span class="fs-5"> [C/C++ API Docs](https://onnxruntime.ai/docs/api/c/){: .btn target="_blank"} </span>|
| <span class="fs-5"> [WinRT API Docs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference){: .btn target="_blank"} </span>|
| <span class="fs-5"> [Objective-C Docs](https://onnxruntime.ai/docs/api/objectivec/index.html){: .btn target="_blank"} </span> |
| <span class="fs-5"> [JavaScript API Docs](https://onnxruntime.ai/docs/api/js/index.html){: .btn target="_blank"} </span>|
| <span class="fs-5"> [Other API Docs](./other-apis){: .btn target="_blank"} </span>|