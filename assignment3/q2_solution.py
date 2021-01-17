"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model
import torch.autograd as autograd

def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
       
   
    t = torch.rand(x.size()[0], 1)
    t = t.expand(x.size())
    x_hat = t * x + ((1 - t) * y)
    x_hat = autograd.Variable(x_hat, requires_grad=True)
    f = critic(x_hat)
    gradients = autograd.grad(outputs=f, inputs=x_hat,
                              grad_outputs=torch.ones(f.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
                         
    fact_grad = gradients.norm(2, dim=1)-1
    t_max = torch.max(torch.FloatTensor([0.]), fact_grad)
    t_max = t_max**2
    
    return torch.mean(t_max)

def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    result = torch.mean(critic(x))-torch.mean(critic(y))
    return result


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. The critic is the
    equivalent of T in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Peason Chi square.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
   
    f1 = 1.0-torch.exp(-critic(x))
    f2 = (1.0-torch.exp(-critic(y)))/torch.exp(-critic(y))
    result = torch.mean(f1)-torch.mean(f2)
    return result

if __name__ == '__main__':
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
