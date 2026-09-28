"""Independent CPU algebra check for the F0 recurrence.

This is a small smooth logistic-regression reference, NOT a reproduction of the
user's Transformer, remote source, or CAL-FIX trajectory. It compares:
  (1) analytic per-step gradient/Hessian + explicit forward state sensitivity;
  (2) an independently written, differentiable optimizer rollout;
  (3) central finite differences of the terminal objective.
No GPU, user checkpoint, private dataset or original research ledger is used.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

DTYPE = torch.float64
T = 256
D = 4
B1, B2, WD = 0.9, 0.999, 0.01
EPS_ADAM, EPS_CLIP, RIDGE = 1e-8, 1e-6, 0.015
N0 = 46

def schedule(phi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    psi = torch.linspace(-1., 1., T, dtype=DTYPE)
    weights = w * torch.exp(phi[1] * psi)
    z = weights.sum() / w.sum()
    return w * torch.exp(phi[0] + phi[1] * psi) / z

def schedule_with_derivative(phi: torch.Tensor, w: torch.Tensor):
    psi = torch.linspace(-1., 1., T, dtype=DTYPE)
    eta = schedule(phi, w)
    weights = w * torch.exp(phi[1] * psi)
    mu = (weights * psi).sum() / weights.sum()
    derivative = torch.stack([eta, eta * (psi - mu)], dim=1)
    return eta, derivative

def data():
    gen = torch.Generator().manual_seed(250925)
    x = torch.randn(T, 12, D, dtype=DTYPE, generator=gen)
    # Deterministically varying batches, with cross-coordinate coupling.
    x = x @ torch.tensor([[1., .3, .2, 0.], [0., .7, -.2, .3],
                          [.1, 0., 1.1, .2], [.2, -.2, .3, .9]], dtype=DTYPE)
    amp = torch.tensor([0.15, 0.7, 2.2, 1.0], dtype=DTYPE)
    x *= amp[torch.arange(T) % 4, None, None]
    teacher = torch.tensor([.7, -.4, .3, -.6], dtype=DTYPE)
    probs = torch.sigmoid(x @ teacher)
    y = (torch.rand(T, 12, dtype=DTYPE, generator=gen) < probs).to(DTYPE)
    qx = torch.randn(25, D, dtype=DTYPE, generator=gen)
    qy = (torch.rand(25, dtype=DTYPE, generator=gen) < torch.sigmoid(qx @ teacher)).to(DTYPE)
    return x, y, qx, qy

def train_loss(theta, x, y):
    logits = x @ theta
    return (torch.nn.functional.softplus(logits) - y*logits).mean() + RIDGE*.5*theta.square().sum()

def terminal_loss(theta, x, y):
    logits = x @ theta
    return (torch.nn.functional.softplus(logits) - y*logits).mean()

def initial():
    return (torch.tensor([.45,-.35,.18,.3], dtype=DTYPE),
            torch.tensor([.008,-.004,.006,-.003], dtype=DTYPE),
            torch.tensor([.025,.032,.02,.028], dtype=DTYPE))

def explicit(phi, w, tau, horizon, x, y, qx, qy):
    eta, deta = schedule_with_derivative(phi, w)
    th, m, v = initial()
    st = torch.zeros(D, 2, dtype=DTYPE)
    sm = torch.zeros_like(st)
    sv = torch.zeros_like(st)
    active = 0
    branch_margin = float('inf')
    for t in range(horizon):
        prob = torch.sigmoid(x[t] @ th)
        g = x[t].T @ (prob-y[t])/len(y[t]) + RIDGE*th
        hessian = x[t].T @ ((prob*(1-prob))[:,None]*x[t])/len(y[t]) + RIDGE*torch.eye(D,dtype=DTYPE)
        hg = hessian @ st
        norm = torch.linalg.vector_norm(g)
        branch_margin = min(branch_margin, abs(float(norm+EPS_CLIP-tau)))
        coef = min(1., float(tau/(norm+EPS_CLIP)))
        c = coef*g
        if coef < 1.:
            active += 1
            # One GLOBAL dot for each sensitivity direction, shared by all blocks.
            global_dot = (g[:,None]*hg).sum(dim=0)
            dc = coef*hg - tau*g[:,None]*global_dot[None,:]/(norm*(norm+EPS_CLIP)**2)
        else:
            dc = hg
        m_next = B1*m + (1-B1)*c
        v_next = B2*v + (1-B2)*c.square()
        sm_next = B1*sm + (1-B1)*dc
        sv_next = B2*sv + 2*(1-B2)*c[:,None]*dc
        bc1 = 1-B1**(N0+t+1)
        bc2 = 1-B2**(N0+t+1)
        mh, vh = m_next/bc1, v_next/bc2
        dmh, dvh = sm_next/bc1, sv_next/bc2
        root = torch.sqrt(vh)
        p = 1/(root+EPS_ADAM)
        dp = -dvh/(2*root[:,None]*(root[:,None]+EPS_ADAM)**2)
        r = p*mh + WD*th
        st_next = st - r[:,None]*deta[t][None,:] - eta[t]*(p[:,None]*dmh+mh[:,None]*dp+WD*st)
        th_next = th-eta[t]*r
        th,m,v,st,sm,sv = th_next,m_next,v_next,st_next,sm_next,sv_next
    query_prob = torch.sigmoid(qx @ th)
    q = qx.T @ (query_prob-qy)/len(qy)
    return {'loss':terminal_loss(th,qx,qy), 'grad':st.T@q, 'theta':th,
            'active_steps':active, 'min_branch_margin':branch_margin}

def rollout_autograd(phi, w, tau, horizon, x, y, qx, qy, differentiate):
    etas = schedule(phi,w)
    th,m,v = initial()
    th = th.requires_grad_(True)
    for t in range(horizon):
        loss = train_loss(th,x[t],y[t])
        g, = torch.autograd.grad(loss,th,create_graph=differentiate)
        # Written as 2 blocks and then reduced as one global norm.
        norm = torch.sqrt(g[:2].square().sum()+g[2:].square().sum())
        c = torch.minimum(torch.ones((),dtype=DTYPE), tau/(norm+EPS_CLIP))*g
        m = B1*m+(1-B1)*c
        v = B2*v+(1-B2)*c*c
        mhat=m/(1-B1**(N0+t+1))
        vhat=v/(1-B2**(N0+t+1))
        # Independent primal spelling, with decoupled weight decay.
        th=th*(1-etas[t]*WD)-etas[t]*mhat/(torch.sqrt(vhat)+EPS_ADAM)
        if not differentiate:
            th=th.detach().requires_grad_(True)
            m=m.detach(); v=v.detach()
    out=terminal_loss(th,qx,qy)
    grad=None
    if differentiate:
        grad,=torch.autograd.grad(out,phi)
    return out.detach(), (grad.detach() if grad is not None else None), th.detach()

def run():
    torch.set_num_threads(1)
    x,y,qx,qy=data()
    grid=torch.linspace(0.,1.,T,dtype=DTYPE)
    w=.008*(1.+.25*torch.cos(torch.pi*grid))
    phi=torch.tensor([.11,-.23],dtype=DTYPE)
    eta,de=schedule_with_derivative(phi,w)
    jad=torch.autograd.functional.jacobian(lambda f:schedule(f,w),phi)
    sched_err=float((de-jad).abs().max())
    assert sched_err < 1e-12
    sum_eta_res=abs(float(eta.sum()-torch.exp(phi[0])*w.sum()))
    sum_b_res=abs(float(de[:,1].sum()))
    assert sum_eta_res < 1e-12 and sum_b_res < 1e-12
    # Adversarial cross-block clipping example.
    gg=torch.tensor([3.,4.],dtype=DTYPE,requires_grad=True)
    hh=torch.tensor([4.,-3.],dtype=DTYPE)
    def clip_fn(v):
        return torch.minimum(torch.ones((),dtype=DTYPE),1/(torch.linalg.vector_norm(v)+EPS_CLIP))*v
    _,truth=torch.autograd.functional.jvp(clip_fn,gg,hh)
    rn=torch.linalg.vector_norm(gg.detach())
    formula=hh/(rn+EPS_CLIP)-gg.detach()*(gg.detach()@hh)/(rn*(rn+EPS_CLIP)**2)
    wrong=hh/(rn+EPS_CLIP)-gg.detach()*(gg.detach()*hh)/(rn*(rn+EPS_CLIP)**2)
    assert float((truth-formula).abs().max()) < 1e-12
    wrong_rel=float(torch.linalg.vector_norm(wrong-truth)/torch.linalg.vector_norm(truth))
    assert wrong_rel > .4
    rows=[]
    for tau in [.02,.25,100.]:
        for horizon in [16,32,256]:
            exp=explicit(phi,w,tau,horizon,x,y,qx,qy)
            pp=phi.clone().requires_grad_(True)
            lv,grad,th=rollout_autograd(pp,w,tau,horizon,x,y,qx,qy,True)
            err=float((exp['grad']-grad).abs().max())
            state_err=float((exp['theta']-th).abs().max())
            fds={}
            for step in [1e-4,1e-5]:
                fd=[]
                for j in range(2):
                    plus=phi.clone(); minus=phi.clone()
                    plus[j]+=step; minus[j]-=step
                    lp,_,_=rollout_autograd(plus,w,tau,horizon,x,y,qx,qy,False)
                    lm,_,_=rollout_autograd(minus,w,tau,horizon,x,y,qx,qy,False)
                    fd.append(float((lp-lm)/(2*step)))
                fds[str(step)]={'grad':fd,'max_abs_error':float((exp['grad']-torch.tensor(fd,dtype=DTYPE)).abs().max())}
            assert err < 1e-10, (tau,horizon,err)
            assert state_err < 1e-11
            assert fds['1e-05']['max_abs_error'] < 1e-7, (tau,horizon,fds)
            rows.append({'clip_tau':tau,'horizon_readout':horizon,'schedule_T':T,
                         'active_steps':exp['active_steps'],'min_branch_margin':exp['min_branch_margin'],
                         'loss':float(lv),'explicit_forward_sensitivity':exp['grad'].tolist(),
                         'full_reverse_autograd':grad.tolist(),'max_abs_forward_reverse_error':err,
                         'max_abs_primal_state_difference':state_err,'finite_differences':fds})
    result={'scope':'Independent synthetic CPU FP64 reference only. NOT remote Transformer validation.',
            'model':'4-parameter coupled smooth logistic loss; analytic Hessian; two parameter blocks; fixed current batches',
            'current_goal':'Check formulas, not certify the original CAL-FIX code or numerical domain.',
            'torch_version':torch.__version__,'schedule_T':T,'beta1':B1,'beta2':B2,'n0':N0,
            'schedule_max_abs_ad_error':sched_err,'sum_eta_identity_abs_error':sum_eta_res,
            'sum_deta_db_abs_error':sum_b_res,
            'clipping_global_jvp':truth.detach().tolist(),'wrong_blockwise_jvp':wrong.detach().tolist(),
            'wrong_blockwise_relative_error':wrong_rel,'cases':rows,
            'max_forward_reverse_abs_error':max(r['max_abs_forward_reverse_error'] for r in rows),
            'max_small_fd_abs_error':max(r['finite_differences']['1e-05']['max_abs_error'] for r in rows),
            'all_assertions_passed':True}
    path=Path(__file__).with_name('independent_cpu_checks.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ['scope','all_assertions_passed','max_forward_reverse_abs_error','max_small_fd_abs_error','wrong_blockwise_relative_error']},ensure_ascii=False,indent=2))
    print('active_steps:',[(r['clip_tau'],r['horizon_readout'],r['active_steps']) for r in rows])
    print('saved:',path)

if __name__=='__main__':
    run()
