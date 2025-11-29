import sys
sys.path.append('build/')
import torch
import time
import math
import agemm  

N, K = 4096, 4096
test_M_pool = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
select_ratio = 0
KE = math.ceil(K * select_ratio / 64) * 64

@torch.no_grad()
def test_quant(M,N,K):

    weight = torch.randn((N, K), dtype=torch.bfloat16).cuda()
    x = torch.randn((M, K), dtype=torch.bfloat16).cuda()
    reorder_index = torch.arange(K, dtype=torch.int16, device='cuda') 

    qw, sfw = agemm.reorder_quantize_w(weight, reorder_index, KE)
    torch.cuda.synchronize()

    elapsed_time_ms = 0
    iterations = 256 * 4096 // M

    for _ in range(iterations):
        torch.cuda.synchronize()
        qx, sfx = agemm.reorder_quantize_x(x, reorder_index, KE)
        torch.cuda.synchronize()

    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # start_event.record()
        qx, sfx = agemm.reorder_quantize_x(x, reorder_index, KE)
        start_event.record()
        y = agemm.matmul(qx, qw, sfx, sfw, 1.0)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / iterations

quant_ms = []
for M in test_M_pool:
    elapsed_time_ms = test_quant(M, N, K)
    quant_ms.append(elapsed_time_ms)

print(quant_ms)