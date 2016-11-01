# module ProximalGradient

using LowRankModels

import LowRankModels: evaluate, grad
evaluate(loss::Loss, X::Array{Float64,2}, w, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w, y) = X'*grad(loss, X*w, y)

export evaluate, grad, proxgrad

function proxgrad(loss, reg, X, y;
                  maxiters = 100,
                  stepsize = 1,
                  w = zeros(size(X,2)),
                  ch = ConvergenceHistory("proxgrad"))
    update_ch!(ch, 0, evaluate(loss, X, w, y) + evaluate(reg, w))
    t = time()
    for i=1:maxiters
        # gradient
        g = grad(loss, X, w, y)
        # prox gradient step
        neww = prox(reg, w - stepsize*g, stepsize)
        # record objective value
        curobj = evaluate(loss, X, neww, y) + evaluate(reg, neww)
        if curobj > ch.objective[end]
          stepsize *= .5
        else
          copy!(w, neww)
          t, told = time(), t
          update_ch!(ch, t - told, curobj)
        end
    end
    return w
end

# end
