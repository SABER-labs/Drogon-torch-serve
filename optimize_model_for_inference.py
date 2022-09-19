import torch
import torchvision
from time import perf_counter
import onnxruntime
model_cpu = torchvision.models.resnet18(pretrained=True)
model_cpu.eval()

model_path = "model_resources/resnet18-v2-7.onnx"

# Export the model to ONNX format
torch.onnx.export(model_cpu,
    torch.randn(1, 3, 224, 224),
    model_path,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input' : {0 : 'batch_size'},
        'output' : {0 : 'batch_size'}
    })

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 5
sess_options.enable_cpu_mem_arena = False
# sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
ort_session = onnxruntime.InferenceSession(model_path, sess_options)


batched_example = torch.randn(32, 3, 224, 224)
single_example = torch.randn(1, 3, 224, 224)
for i in range(4):
    model_cpu(batched_example)
start = perf_counter()
model_cpu(batched_example)
py_time_taken = perf_counter() - start
print(f"PyTorch CPU inference time for {batched_example.size(0)} batch_size was : {py_time_taken*1000:.2f}ms")
print(f"PyTorch CPU inference time for {1} batch_size in batched_mode was : {py_time_taken * 1000 / batched_example.size(0):.2f}ms")
start = perf_counter()
model_cpu(single_example)
print(f"PyTorch CPU inference time for {single_example.size(0)} batch_size was : {(perf_counter() - start)*1000:.2f}ms")

ort_inputs = {'input': batched_example.detach().numpy()}
for i in range(4):
    ort_outs = ort_session.run(None, ort_inputs)
start = perf_counter()
ort_outs = ort_session.run(None, ort_inputs)
ort_time_taken = perf_counter() - start
print(f"ONNX CPU inference time for {batched_example.size(0)} batch_size was : {ort_time_taken * 1000:.2f}ms")
print(f"ONNX CPU inference time for {1} batch_size in batched_mode was : {ort_time_taken * 1000 / batched_example.size(0):.2f}ms")
start = perf_counter()
ort_session.run(None, {'input': single_example.detach().numpy()})
print(f"ONNX CPU inference time for {single_example.size(0)} batch_size was : {(perf_counter() - start) * 1000:.2f}ms")
print(f"ONNX is faster than Pytorch model by: {(1 - (ort_time_taken/py_time_taken)) * 100:.2f}%")
