using RDatasets
using Random
using Statistics
using LinearAlgebra

function split(data, labels; dims=1)
    n = length(labels)
    size(data, dims) == n || throw(DimensionMismatch("..."))

    ratio_train = 0.8
    n_train = round(Int, ratio_train*n) # 0.8 * number of "pictures"
    i_rand = randperm(n) # construct a random permutation of length n
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
norm(aux1[1] - aux2[1]') # => 1.4512633279159294e-14