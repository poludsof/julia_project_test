using Flux
using MLDatasets
using Random
using JuMP
using GLPK
using Plots
using Flux: params
using StatsBase

Random.seed!(666)

# X_train_to_test, y_train_to_test = MLDatasets.MNIST.traindata()
X_train, y_train = MLDatasets.MNIST.traindata()
X_test, y_test = MLDatasets.MNIST.testdata()
X_train = Float32.(X_train)
y_train = Flux.onehotbatch(y_train, 0:9)

# random_indices = sample(1:size(X_train, 2), 200, replace=false)
# X_train_small = X_train[:, random_indices]
# y_train_small = y_train[:, random_indices]

n_hidden = 256
n_classes = 10

model = Chain(
    Dense(784, n_hidden, relu),
    Dense(n_hidden, n_classes, relu), softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(0.0001)

parameters = params(model)
# flatten() function converts array 28x28x60000 into 784x60000
train_data = [(Flux.flatten(X_train), Flux.flatten(y_train))]
for i in 1:50
    Flux.train!(loss, parameters, train_data, optimizer)
end

function accuracy(x_test, y_test)
    test_data = [(Flux.flatten(x_test), y_test)]
    accuracy = 0
    for i in 1:length(y_test)
        if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
            accuracy = accuracy + 1
        end
    end
    return accuracy / length(y_test)
end

accuracy(X_test, y_test)

function interpret_pixels_with_milp(model, input_image)
    milp_model = Model(GLPK.Optimizer)

    W1 = params(model[1])[1]
    b1 = params(model[1])[2]

    h = @variable(milp_model, [1:n_hidden])
    @constraint(milp_model, [j=1:n_hidden], h[j] == sum(W1[j, i] * input_image[i] for i=1:28*28) + b1[j])
    @constraint(milp_model, [j=1:n_hidden], h[j] >= 0)
    @objective(milp_model, Max, h[1])

    optimize!(milp_model)
    hidden_activations = value.(h)
    return hidden_activations
end

example_input = X_train[:, :, 5]

activations = interpret_pixels_with_milp(model, example_input)
println("Activations: ", activations)

function visualize_pixel_importance(input_image, activations)
    normalized_activations = activations ./ maximum(abs.(activations))
    pixel_importance = input_image .* sum(normalized_activations)
    importance_reshaped = reshape(pixel_importance, 28, 28)
    heatmap(importance_reshaped, c=:hot, xlabel="X", ylabel="Y", title="Pixel Importance via MILP")
end
visualize_pixel_importance(example_input, activations)