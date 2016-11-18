# module ProximalGradient

using LowRankModels

import LowRankModels: evaluate, grad
evaluate(loss::Loss, X::Array{Float64,2}, w::Array{Float64,1}, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w::Array{Float64,1}, y) = X'*grad(loss, X*w, y)
evaluate(loss::Loss, X::Array{Float64,2}, w::Array{Float64,2}, y) = evaluate(loss, X*w, y)
grad(loss::Loss, X::Array{Float64,2}, w::Array{Float64,2}, y) = X'*grad(loss, X*w, y)

export evaluate, grad, proxgrad, is_differentiable

is_differentiable(l::QuadLoss) = true
is_differentiable(l::L1Loss) = false
is_differentiable(l::HuberLoss) = true
is_differentiable(l::QuantileLoss) = false
is_differentiable(l::PoissonLoss) = true
is_differentiable(l::WeightedHingeLoss) = false
is_differentiable(l::LogisticLoss) = true
is_differentiable(l::OrdinalHingeLoss) = false
is_differentiable(l::OrdisticLoss) = true
is_differentiable(l::MultinomialOrdinalLoss) = true
is_differentiable(l::BvSLoss) = is_differentiable(l.bin_loss)
is_differentiable(l::MultinomialLoss) = true
is_differentiable(l::OvALoss) = is_differentiable(l.bin_loss)
is_differentiable(l::PeriodicLoss) = true

function proxgrad(loss::Loss, args...; kwargs...)
  return proxgrad_linesearch(loss, args...; kwargs...)
  # if is_differentiable(loss)
  #   return proxgrad_linesearch(loss, args...; kwargs...)
  # else
  #   return proxgrad_dec(loss, args...; kwargs...)
  # end
end

function proxgrad_linesearch(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1,
                  w = (embedding_dim(loss)==1 ? zeros(size(X,2)) : zeros(size(X,2), embedding_dim(loss))),
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

function proxgrad_dec(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1,
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

function proxgrad_const(loss::Loss, reg::Regularizer, X::Array{Float64,2}, y;
                  maxiters = 100,
                  stepsize = 1,
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
