import torch
from tqdm import trange
import renderer
def motion_Nlength(data,N,overlap=True):
    results = [] 
    out =[]
    for idx,motion in enumerate(data):
        result = []
        
        for parts in motion:
            result.append(parts)
#         result = result + [[0.,0.] for i in range(12)]
        out.append(result)
        
        if overlap==True and idx>=N-1:
#             print(torch.tensor(out[idx-N+1:idx+1])
            results.append(torch.tensor(out[idx-N+1:idx+1]).permute(1,2,0))
            
        elif (idx+1)%N==0:
            results.append(torch.tensor(out[idx-N+1:idx+1]))
    return results


def sample(diffusion_model,n_samples=64,device='None',n_steps=1000,N=32):
    """
    ### Sample images
    """
    all_x = []
    with torch.no_grad():
        x = torch.randn([n_samples,20,2,N],
                        device=device)

        # Remove noise for $T$ steps
        for t_ in trange(n_steps):
            
            t = n_steps - t_ - 1
            x = diffusion_model.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

        return x



def kps_generate(cols,x):
    '''x: shape(20,2,N)'''
    x = x.permute(2,0,1)
    result = {}
    for col in cols:
        result[col]=[]
    for item in x:
        y = (item).tolist() 
        for idx, col in enumerate(cols):
            result[col].append(y[idx])
    return result