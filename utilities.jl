using RDatasets
using Random
using Statistics
using LinearAlgebra

iris = dataset("datasets", "iris")

X = Matrix(iris[:, 1:4])
y = iris.Species

function split(X, y::AbstractVector; dims=1, ratio_train=0.8, kwargs...)
    n = length(y)
    size(X, dims) == n || throw(DimensionMismatch("..."))

    n_train = round(Int, ratio_train*n) # 0.8 * number of "pictures"
    i_rand = randperm(n)  # construct a random permutation of length n
    i_train = i_rand[1:n_train] # cut the index array by the number n_train
    i_test = i_rand[n_train+1:end]

    return selectdim(X, dims, i_train), y[i_train], selectdim(X, dims, i_test), y[i_test]
end

function normalize(X_train, X_test; dims=1, kwargs...)
    col_mean = mean(X_train; dims)
    col_std = std(X_train; dims)
    return (X_train .- col_mean) ./ col_std, (X_test .- col_mean) ./ col_std
end

function onehot(y, classes)
    y_onehot = zeros(length(classes), length(y))
    for (idx, class) in enumerate(classes)
        y_onehot[idx, y .== class] .= 1  # element-by-element comparison .== and assignment .=
    end  
    # println("onehot:", y_onehot[1:10])
    return y_onehot
end

onecold(y, classes) = [classes[argmax(y_col)] for y_col in eachcol(y)]

iris = dataset("datasets", "iris")

X = Matrix(iris[:, 1:4])
y = iris.Species

# test onehot and onecold functions
classes = unique(y)
isequal(onecold(onehot(y, classes), classes), y)
# => true


# combine functions above into one function
function prepare_data(X, y; do_normal=true, do_onehot=true, kwargs...)
    X_train, y_train, X_test, y_test = split(X, y; kwargs...)

    if do_normal
        X_train, X_test = normalize(X_train, X_test; kwargs...)
    end

    classes = unique(y)

    if do_onehot
        y_train = onehot(y_train, classes)
        y_test = onehot(y_test, classes)
    end

    return X_train, y_train, X_test, y_test, classes
end

Random.seed!(666)

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

X_train, y_train, X_test, y_test, classes = prepare_data(X', y; dims=2)


Random.seed!(666)
aux1 = prepare_data(X, y; dims=1)
Random.seed!(666)
aux2 = prepare_data(X', y; dims=2)
# norm(aux1[1] - aux2[1]') # => 1.4512633279159294e-14

# Create NN with 3 layers
struct SimpleNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

# Constructor
SimpleNet(n1, n2, n3) = SimpleNet(randn(n2, n1), randn(n2), randn(n3, n2), randn(n3))

Random.seed!(666)

# size(X_train, 1) - number of rows in X_train
m = SimpleNet(size(X_train, 1), 5, size(y_train, 1))

# m is a SimpleNet x is an argument
# z1 = W1*x .+ b1
function (m::SimpleNet)(x) 
    z1 = m.W1*x .+ m.b1 # layer 1
    a1 = max.(z1, 0) # RELU
    z2 = m.W2*a1 .+ m.b2 # layer 2
    p = exp.(z2) ./ sum(exp.(z2), dims=1) # softmax
    return p
end

m(X_train[:,1:2])

function grad(m::SimpleNet, x::AbstractVector, y; ϵ=1e-10)
    z1 = m.W1*x .+ m.b1
    a1 = max.(z1, 0)
    z2 = m.W2*a1 .+ m.b2
    a2 = exp.(z2) ./ sum(exp.(z2), dims=1)
    l = -sum(y .* log.(a2 .+ ϵ)) # loss function

    e_z2 = exp.(z2)
    l_part = (- e_z2 * e_z2' + Diagonal(e_z2 .* sum(e_z2))) / sum(e_z2)^2

    l_a2 = - y ./ (a2 .+ ϵ)
    l_z2 = l_part * l_a2
    l_a1 = m.W2' * l_z2
    l_z1 = l_a1 .* (a1 .> 0)
    l_x = m.W1' * l_z1

    l_W2 = l_z2 * a1'
    l_b2 = l_z2
    l_W1 = l_z1 * x'
    l_b1 = l_z1

    return l, l_W1, l_b1, l_W2, l_b2
end

# compute gradients of multiple samples
g_all = [grad(m, X_train[:,k], y_train[:,k]) for k in 1:size(X_train,2)]
typeof(g_all)

# mean over all samples
mean_tuple(d::AbstractArray{<:Tuple}) = Tuple([mean([d[k][i] for k in 1:length(d)]) for i in 1:length(d[1])])

function first_train()
    α = 1e-1
    max_iter = 200
    L = zeros(max_iter)
    for iter in 1:max_iter
        grad_all = [grad(m, X_train[:,k], y_train[:,k]) for k in 1:size(X_train,2)]
        grad_mean = mean_tuple(grad_all)

        L[iter] = grad_mean[1]

        m.W1 .-= α*grad_mean[2]
        m.b1 .-= α*grad_mean[3]
        m.W2 .-= α*grad_mean[4]
        m.b2 .-= α*grad_mean[5]
    end
end

predict(X) = m(X)
accuracy(X, y) = mean(onecold(predict(X), classes) .== onecold(y, classes))

# println("Train accuracy = ", accuracy(X_train, y_train))
# println("Test accuracy = ", accuracy(X_test, y_test))