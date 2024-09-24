import torch
import torch.nn.functional as F
import numpy as np
import math


# https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def DLG(net, origin_grad, target_inputs):
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # generate dummy data and label
        dummy_data = torch.randn_like(gt_data, requires_grad=True)
        dummy_out = torch.randn_like(gt_out, requires_grad=True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_out])

        history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():
                optimizer.zero_grad()

                dummy_pred = net(F.sigmoid(dummy_data))
                dummy_loss = criterion(dummy_pred, dummy_out)
                dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad): 
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)

        history.append(F.sigmoid(dummy_data).data.cpu().numpy())
        
        p = psnr(history[0], history[2])
        if not math.isnan(p):
            psnr_val += p
            cnt += 1

    if cnt > 0:
        return psnr_val / cnt
    else:
        return None