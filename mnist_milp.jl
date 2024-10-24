using JuMP
using MLDatasets
using LinearAlgebra
using GLPK

X_train, y_train = MNIST(split=:train)[:]
X_test, y_test = MNIST(split=:test)[:]
# X_test, y_test = MNIST.testdata() # MNIST.testdata() is deprecated, use `MNIST(split=:test)[:]` instead

X_train = reshape(X_train, 784, :)'
X_test = reshape(X_test, 784, :)'

model = Model(GLPK.Optimizer)

n_samples, n_features = size(X_train)
n_classes = 10

n_samples = 10000
X_train_subset = X_train[1:n_samples, :]
y_train_subset = y_train[1:n_samples]

@variable(model, W[1:n_features, 1:n_classes])
@variable(model, b[1:n_classes])
@variable(model, z[1:n_samples, 1:n_classes], Bin)

M = 1e6

@constraint(model, [i=1:n_samples], sum(z[i, :]) == 1)
@constraint(model, [i=1:n_samples, k=1:n_classes, l=1:n_classes; k != l], 
    sum(X_train[i, j] * W[j, k] for j in 1:n_features) + b[k] >= 
    sum(X_train[i, j] * W[j, l] for j in 1:n_features) + b[l] - M * (1 - z[i, k])
)

@objective(model, Min, sum((1 - z[i, y_train_subset[i] + 1]) for i in 1:n_samples))
optimize!(model)
W_optimal = value.(W)
b_optimal = value.(b)
z_optimal = value.(z)

function predict(X, W_optimal, b_optimal)
    predictions = []
    for i in 1:size(X, 1)
        scores = W_optimal' * X[i, :] .+ b_optimal  # X[i, :] is 1×784, W_optimal is 784×10, result is 1×10
        push!(predictions, argmax(scores) - 1)  # Subtract 1 to match the class labels (0-9)
    end
    return predictions
end

y_pred = predict(X_train_subset, W_optimal, b_optimal)
accuracy = sum(y_pred .== y_train_subset) / length(y_train_subset)
println("Accuracy on test set: ", accuracy * 100, "%")