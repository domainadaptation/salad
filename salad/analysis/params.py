param_collection = [{}, {}, {}, {}]

N = 9

n_models = len(fnames)

for n, fname in enumerate(fnames):
    model = torch.load(fname)
    
    print(n, len(param_collection))

    for i in range(N):

        for j, layer in enumerate(model.conditional_layers):
            
            vals = get_transform(layer.layers[i])
            
            for val, param_dict in zip(vals, param_collection): 
                
                val = val.data.detach().cpu().numpy()

                p             = param_dict.get(j, np.zeros((10,n_models) + val.shape))
                p[i,n]        = val
                param_dict[j] = p

P = []
for p in param_collection:
    params = [p[i] for i in range(len(p))]
    P.append(np.concatenate(params, axis=-1))
P = np.stack(P, axis=0)[:,:9]
P.shape

def compute_linear(params):

    mu, var, gamma, beta = params

    inv_std = (1e-5 + var)**(-.5)

    b = beta - (mu * gamma) * inv_std 
    m = gamma * inv_std

    return m, b

def get_transform(bn_layer):
    
    mu    = bn_layer.running_mean
    var   = bn_layer.running_var
    gamma = bn_layer.weight
    beta  = bn_layer.bias
    
    inv_std = (1e-5 + var)**(-.5)
    
    b = beta - (mu * gamma) * inv_std 
    m = gamma * inv_std
    
    m = m.data.detach().cpu().numpy()
    b = b.data.detach().cpu().numpy()
    
    return mu, var, gamma, beta