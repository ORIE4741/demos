# module ProximalGradient

using LowRankModels

import LowRankModels: evaluate, grad
evaluate(loss::Loss, X::Array{Float64,2}, w::Array{Float64,1}, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w::Array{Float64,1}, y) = X'*grad(loss, X*w, y)
evaluate(loss::Loss, X::Array{Float64,2}, w::Array{Float64,2}, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w::Array{Float64,2}, y) = X'*grad(loss, X*w, y)

export evaluate, grad, proxgrad

"""proximal gradient with a linesearch that falls back to a constant (smaller) stepsize
when no descent direction is found"""
function proxgrad(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1,
                  c = .1, # sufficient decrease
                  max_inner_iters = 100, # if stepsize decreases 100 times, assume problem is actually nonsmooth
                  w = (embedding_dim(loss)==1 ? zeros(size(X,2)) : zeros(size(X,2), embedding_dim(loss))),
                  ch = ConvergenceHistory("proxgrad"))
    oldloss = evaluate(loss, X, w, y)
    update_ch!(ch, 0, oldloss + evaluate(reg, w))
    t = time()
    stepsize0 = copy(stepsize)
    for i=1:maxiters
        # gradient
        g = grad(loss, X, w, y)
        # prox gradient step
        grad_step = w - stepsize*g
        w_new = prox(reg, grad_step, stepsize)

        # linesearch 
        curloss = evaluate(loss, X, w_new, y) # record loss value
        inner_iters = 0 # just in case
        while curloss - oldloss >= -c*stepsize*dot(g, w_new - w) # gradient approximation is not good
            stepsize /= 2
            grad_step = w - stepsize*g
            w_new = prox(reg, grad_step, stepsize)
            curloss = evaluate(loss, X, w_new, y)
            inner_iters +=1
            if inner_iters > max_inner_iters # nonsmooth; fall back to constant stepsize
                return proxgrad_const(loss, reg, X, y, maxiters=maxiters, stepsize = stepsize0/10, w = w, ch = ch)
            end
        end
        # take the step
        copy!(w, w_new)
        oldloss = curloss
        #stepsize = copy(stepsize0)
        # record objective value and elapsed time
        t, told = time(), t
        update_ch!(ch, t - told, curloss + evaluate(reg, w))
    end
    return w
end

"""proximal gradient method with Armijo-Wolfe linesearch"""
function proxgrad_linesearch(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1.,
                  c = .1, # sufficient decrease
                  w = (embedding_dim(loss)==1 ? zeros(size(X,2)) : zeros(size(X,2), embedding_dim(loss))),
                  ch = ConvergenceHistory("proxgrad"))
    oldloss = evaluate(loss, X, w, y)
    update_ch!(ch, 0, oldloss + evaluate(reg, w))
    t = time()
    for i=1:maxiters
        # gradient
        g = grad(loss, X, w, y)
        # prox gradient step
        grad_step = w - stepsize*g
        w_new = prox(reg, grad_step, stepsize)
        # record loss value
        curloss = evaluate(loss, X, w_new, y)
        while curloss - oldloss >= -c*stepsize*dot(g, w_new - w) # gradient approximation is not good
            stepsize /= 2
            grad_step = w - stepsize*g
            w_new = prox(reg, grad_step, stepsize)
            curloss = evaluate(loss, X, w_new, y)
        end
        # take the step
        copy!(w, w_new)
        oldloss = curloss
        # record objective value and elapsed time
        t, told = time(), t
        update_ch!(ch, t - told, curloss + evaluate(reg, w))
    end
    return w
end

"""proximal gradient method with decreasing stepsize"""
function proxgrad_dec(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1.,
                  w = (embedding_dim(loss)==1 ? zeros(size(X,2)) : zeros(size(X,2), embedding_dim(loss))),
                  ch = ConvergenceHistory("proxgrad"),
                  verbose = true)
    wbest = copy(w)
    update_ch!(ch, 0, evaluate(loss, X, w, y) + evaluate(reg, w))
    t = time()
    if verbose
      println("using decreasing stepsize for nondifferentiable loss")
    end
    for i=1:maxiters
        # gradient
        g = grad(loss, X, w, y)
        # prox gradient step
        w = prox(reg, w - stepsize/i*g, stepsize/i)
        # record objective value
        obj = evaluate(loss, X, w, y) + evaluate(reg, w)
        if obj < ch.objective[end]
          if verbose
            println("found a better obj $obj")
          end
          copy!(wbest, w)
          update_ch!(ch, time() - t, obj)
        end
    end
    return wbest
end

"""proximal gradient method with constant stepsize"""
function proxgrad_const(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1.,
                  w = (embedding_dim(loss)==1 ? zeros(size(X,2)) : zeros(size(X,2), embedding_dim(loss))),
                  ch = ConvergenceHistory("proxgrad"))
    wbest = copy(w)
    update_ch!(ch, 0, evaluate(loss, X, w, y) + evaluate(reg, w))
    t = time()
    for i=1:maxiters
        # gradient
        g = grad(loss, X, w, y)
        # prox gradient step
        w = prox(reg, w - stepsize*g, stepsize)
        # record objective value
        obj = evaluate(loss, X, w, y) + evaluate(reg, w)
        if obj < ch.objective[end]
          copy!(wbest, w)
          update_ch!(ch, time() - t, obj)
        end    end
    return wbest
end


# end
