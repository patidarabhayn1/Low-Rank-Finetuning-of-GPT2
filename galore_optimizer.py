import torch
from torch.optim import Optimizer


class G_ADAM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, rank = 4, update_interval = 10, device = 'cuda:4'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(G_ADAM, self).__init__(params, defaults)
        self.rank = rank
        self.update_interval = update_interval
        self.device = device

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]

                apply_galore = True
                if len(p.grad.data.shape) == 1:
                    apply_galore = False

                if apply_galore:
                    if len(state) == 0:
                        state['step'] = 0
                        U, _, _ = torch.svd(p.grad.data)
                        state['U'] = U[:, :self.rank]
                        state['m'] = torch.zeros((min(self.rank, p.grad.data.shape[0]), p.grad.data.shape[1])).to(self.device)
                        state['v'] = torch.zeros((min(self.rank, p.grad.data.shape[0]), p.grad.data.shape[1])).to(self.device)
                    
                    if state['step'] % self.update_interval == 0:
                        U, _, _ = torch.svd(p.grad.data)
                        state['U'] = U[:, :self.rank]
                else:
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                if apply_galore:
                    grad = torch.mm(state['U'].t(), grad).to(self.device)

                state['step'] += 1
                # print("M: ", m.shape)
                # print("V: ", v.shape)
                # print("Grad: ", grad.shape)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                update = group['lr'] * m_hat / (v_hat ** 0.5 + group['eps'])
                if apply_galore:
                    update = torch.mm(state['U'], update)

                p.data.add_(-update)

        return loss