# input X_train, Y_train, lr, lambda: L2, X_min, X_max,Y_min, Y_max,iter,loss function
# set min,max to none if not available

import numpy as np

def grad(x:np.array,y:np.array,w:np.array,b:float,lambda_:float,n:int):
    #print('grad')
    #calculate grad
    pred = np.dot(x,w) + b                  
    e = pred - y                  
    dc_dw = np.dot(x.T,e) / n  + (lambda_ / n) * w
    dc_db = np.sum(e,axis=0) / n       

    return dc_dw,dc_db

def adam(param, grad, m, v, iter, lr, beta1, beta2, epsilon):
    #print('reched adam')
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** iter)
    v_hat = v / (1 - beta2 ** iter)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def MSE(pred,Y,n_samp,lambda_,w):
    reg_term = (lambda_ / (2 * n_samp)) * np.sum(w ** 2)     #only L2 reg.
    cost = (np.sum((pred - Y) ** 2)/n_samp)+reg_term    # mse cost

    return cost

def Cross_entropy():

    return

loss_functions = {
    'MSE': MSE,
    'Cross_entropy': Cross_entropy,
    # add more if needed
}
def linearRegression(X: np.array, Y: np.array, lr: float, lambda_: float,X_min:np.array,X_max:np.array,Y_min:np.array,Y_max:np.array,iter:int,loss:str):
    """
    Parameters:
    - X: Input feature matrix (NumPy array)
    - Y: Target vector (NumPy array)
    - lr: Learning rate (float)
    - lambda_: L2 regularization coefficient (float)

    Returns:
    - weights: Learned model parameters
    """

    loss_fn = loss_functions[loss]
    if X_min is None:
        X_min = X.min(axis=0)
    if X_max is None:
        X_max = X.max(axis=0)
    if Y_min is None:
        Y_min = Y.min(axis=0) if Y.ndim > 1 else Y.min()
    if Y_max is None:
        Y_max = Y.max(axis=0) if Y.ndim > 1 else Y.max()
    # assuming X,Y are multi dimensional
    n_samples,n_features=X.shape
    m_out,out_dim=Y.shape if len(Y.shape) > 1 else (n_samples, 1)
    w = 0.01 * np.random.randn(n_features, out_dim)
    print(w.shape)
    b = 0.01 * np.random.randn(out_dim)
    print(b.shape)
    #print('reched ini')
    #momentum
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)

    cost_hist=[]
    w_hist=[]
    b_hist=[]
    
    X_norm= (X - X_min) / (X_max - X_min + 1e-6)
    Y_norm= (Y - Y_min) / (Y_max - Y_min+ 1e-6)
    
    prev_cost= float('inf')
    for i in range(iter):
        #calculate cost
        #print('reched iter')
        pred = np.dot(X_norm,w) + b   
        cost = loss_fn(pred=pred,Y=Y_norm,n_samp=n_samples,lambda_=lambda_,w=w)

        #update
        dc_dw,dc_db= grad(X_norm,Y_norm,w,b,lambda_,n_samples)
        w, m_w, v_w = adam(w, dc_dw, m_w, v_w, i+1, lr, 0.9, 0.999, 1e-8)
        b, m_b, v_b = adam(b, dc_db, m_b, v_b, i+1, lr, 0.9, 0.999, 1e-8)

        cost_hist.append(cost)
        w_hist.append(w)
        b_hist.append(b)
        

        if abs(prev_cost-cost)< 1e-8:
            print(f"Converged at iteration {i}")
            print(f" final Cost {float(cost_hist[-1]):8.2f} ")
            break
        prev_cost= cost
        if i%10 ==0:
            print(f"Iteration {i:4}: Cost {float(cost_hist[-1]):8.2f} ")

    return w,b,cost_hist,w_hist,b_hist

    
