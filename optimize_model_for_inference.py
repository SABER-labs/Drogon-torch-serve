import torch
import torchvision

model_cpu = torchvision.models.resnet18(pretrained=True)
model_cpu.eval()
traced_model_cpu = torch.jit.script(model_cpu)
# frozen_traced_model_cpu = torch.jit.freeze(traced_model_cpu)
# optimized_frozen_traced_model_cpu = torch.jit.optimize_for_inference(traced_model_cpu)
traced_model_cpu.save("model_resources/resnet18_traced_cpu.pt")

model_cuda = torchvision.models.resnet18(pretrained=True).cuda().half()
model_cuda.eval()
traced_model_cuda = torch.jit.script(model_cuda)
frozen_traced_model_cuda = torch.jit.freeze(traced_model_cuda)
optimized_frozen_traced_model_cuda = torch.jit.optimize_for_inference(frozen_traced_model_cuda)
optimized_frozen_traced_model_cuda.save("model_resources/resnet18_traced_cuda.pt")
