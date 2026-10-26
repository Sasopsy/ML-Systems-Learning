import torch,json
from pathlib import Path
p=Path('project_03_gpu_operator_library/results/2026-10-27-pytorch-implementation')
x=((torch.arange(4096*1024,device='cuda')%41-20)*0.25).float().reshape(4096,1024)
y=torch.empty_like(x)
results={}
with torch.no_grad():
 for name in ['default','preallocated']:
  def run():
   return torch.softmax(x,dim=-1) if name=='default' else torch.softmax(x,dim=-1,out=y)
  for _ in range(50):run()
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   run();torch.cuda.synchronize()
  prof.export_chrome_trace(str(p/(name+'.trace.json')))
  results[name]=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
results.update(version=torch.__version__,git_revision=torch.version.git_version,shape=list(x.shape),dtype=str(x.dtype))
(p/'dispatch.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
