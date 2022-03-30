#¤ Equations to integrate:

f1(x) =        cos(x^2) * exp(-x)
f2(x) =   √x * cos(x^2) * exp(-x)
f3(x) = 1/√x * cos(x^2) * exp(-x)
f4(x) = 1000*exp(-1/x)*exp(-1/(1-x))

##

using GLMakie; GLMakie.activate!()
plot(0..1, f1)
plot(0..1, f2)
plot(0..1, f3)
plot(0..1, f4)
##

##
#¤ For getting \alpha=2:
function is_power_of_2(x::Int)
    current_guess = 1
    while x>current_guess
        current_guess *=2
    end
    return current_guess == x ? true : false
end

function integrate(N::Int, f::Function, method::Symbol; interval=(0, 1), return_f_evals = false)
    my_xs = range(interval..., N)
    h = my_xs |> step 
    if method == :rectangle
        x_midpoints = my_xs[begin:end-1] .+ h/2
        f_evals = length(x_midpoints)
        A = h * sum(f.(x_midpoints))
        return_f_evals ? (return A, f_evals) : return A
        
    elseif method == :trapezoid
        f_evals = length(my_xs)
        A = h * sum([1//2*f(my_xs[begin]); f.(my_xs[begin+1:end-1]); 1//2*f(my_xs[end])])
        return_f_evals ? (return A, f_evals) : return A

    elseif method == :simpsons
        @assert isodd(N) && N≥5  "N (=$N) is not odd and ≥ 5, which is assumed in the implementation."
        f_evals = length(my_xs)
        endpoints = my_xs[[begin, end]]  .|> f
        midpoints = my_xs[begin+1:2:end-1] .|> f
        internal_edgepoint = my_xs[begin+2:2:end-2] .|> f
        weighted_points = vcat(endpoints .* 1/3, midpoints .* 4/3, internal_edgepoint .* 2/3)
        A = h * sum(weighted_points)
        return_f_evals ? (return A, f_evals) : return A

    else
        "Method not recognized" |> error
    end
end
N_temp = 1000 * 2 + 1
integrate(N_temp, f1, :rectangle)
integrate(N_temp, f1, :trapezoid)
integrate(N_temp, f1, :simpsons)


integrate(2^16+1, f2, :simpsons)

integrate(2^16+1, f3, :rectangle)

integrate(2^16+1, f2, :trapezoid)


using DataFrames
function make_table(f, Ns, method)   
    global results = DataFrame(N=Int[], A=[], ΔA=[], αᵏ=[], RichError=[], f_evals=Int[])
    push!(results, [0; fill(NaN, 4); 0])
    for N in Ns
        @assert is_power_of_2(N-1) "The number of intervals is not a power of 2, which is assumed in the implementation."
        A, f_evals = integrate(N, f, method, return_f_evals=true)
        ΔA = A-results.A[end]
        αᵏ = results.ΔA[end]/ΔA
        RichError = ΔA/(αᵏ-1)
        push!(results, [N, A, ΔA, αᵏ, RichError, f_evals])
    end
    return results[begin+1:end, :]
end

function Ns_range(N, start_power=0, α=2)
    if α == 2
        N ≥ 64  &&  @warn "You are possibly going higher than typemax(Int64)."
    else
        @warn "Not checking overflow."
    end
    return 1 .+ α .^ range(start_power, length=N)
end
Ns_range(64)
#! αᵏ goes to 2 for trapezoid...
let Ns = Ns_range(10)
    method = :trapezoid
    make_table(f1, Ns, method) |> println
end

##

plot(0..1, f1)
let Ns = Ns_range(10)
    method = :rectangle
    make_table(f1, Ns, method) |> println
end

let Ns = Ns_range(15)
    method = :trapezoid
    make_table(f1, Ns, method) |> println
end

let Ns = Ns_range(10, 2)
    method = :simpsons
    make_table(f1, Ns, method) |> println
end


plot(0..1, f2)
let Ns = Ns_range(10, 2)
    method = :simpsons
    make_table(f2, Ns, method) |> println
end

plot(0..1, f3)
let Ns = Ns_range(25) #! With 16_777_217 points, still high error...
    method = :rectangle
    make_table(f3, Ns, method) |> println
end

plot(0..1, f4)
let Ns = Ns_range(10) #! All funky αᵏ, and therefor RichError
    method = :trapezoid
    make_table(f4, Ns, method) |> println
end

#¤ If the order is not well defined for the problem, 
#¤ assuming that the error is at all related to 
#¤ the RichError is not valid.

#¤ "If you don't have singularities, the order will be the 
#¤ expected order"

#¤ "If you can not get an order out of the problem, 
#¤ you should apply a doubble exponential.


##! 30/03
#¤ DE rule - double exporential rule error estimate is inaccuracy, 
#¤ but fast convergence. "Formally can not estimate error by richardson"

#¤ To prevent problems with singularities and computer precision, 
#¤ we can define f(x(t)) as F(t), circumventing x and underflow
#? where a and b are the integration interval
f(x) = something
q(t) = exp(-2*sinh(t))
d(t) = (b-a) * q(t)/(1+q(t))
F(t) = t<0 ? f(a+d(-t)) : f(b-d(t))
∫ₐᵇ f(x) dx = ∫_-∞ ^∞ f(x(t)) dx/dt dt= ∫_-∞ ^∞ F(x) dx/dt dt