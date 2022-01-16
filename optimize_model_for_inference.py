import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
model.eval()
traced_model_cpu = torch.jit.script(model)
frozen_traced_model_cpu = torch.jit.freeze(traced_model_cpu)
optimized_frozen_traced_model_cpu = torch.jit.optimize_for_inference(frozen_traced_model_cpu)
optimized_frozen_traced_model_cpu.save("model_resources/resnet18_traced_cpu.pt")

model = model.cuda().half()
traced_model_cuda = torch.jit.script(model)
frozen_traced_model_cuda = torch.jit.freeze(traced_model_cuda)
optimized_frozen_traced_model_cuda = torch.jit.optimize_for_inference(frozen_traced_model_cuda)
optimized_frozen_traced_model_cuda.save("model_resources/resnet18_traced_cuda.pt")
