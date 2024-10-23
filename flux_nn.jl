include(raw"utilities.jl")
using Flux

using RDatasets
using Random

Random.seed!(666)
iris = dataset("datasets", "iris")

X = Matrix(iris[:, 1:4])
y = iris.Species

X_train, y_train, X_test, y_test, classes = prepare_data(X', y; dims=2)

n_hidden = 5
# model
m = Chain( # Chain is used to group a sequence of layers into a single object that can be used to train the neural network.
    # Dense(number of input neurons, number of neurons in this layer, activation function)
    Dense(size(X_train, 1), n_hidden, relu),
    Dense(n_hidden, size(y_train,1), identity),
    softmax,
)

m(X_train)

using Flux: params
# We can select the second layer of m by m[2]
# Since the second layer has 5 input and 3 output neurons, 
# its parameters are a matrix of size 3×5 and a vector of length 3
# The parameters params(m[2]) are a tuple of (the matrix and the vector).
params(m[2])[2] .= [-1;0;1]

using Flux: crossentropy
L(y_pred,y_true) = crossentropy(y_pred, y_true) # define loss function
L(m(X_train), y_train) # => 1.5389597f0

# The function gradient takes two inputs.
# The first one is the function we want to differentiate, and the second one are the parameters.
grads = Flux.gradient(m -> L(m(X_train), y_train), m)

# Train the classifiers for 250 iterations
opt = Descent(0.1)
opt_state = Flux.setup(opt, m)
max_iter = 250

acc_train = zeros(max_iter)
acc_test = zeros(max_iter)
for i in 1:max_iter
    gs = Flux.gradient(m -> L(m(X_train), y_train), m)
    Flux.update!(opt_state, m, gs[1])
    acc_train[i] = accuracy(X_train, y_train)
    acc_test[i] = accuracy(X_test, y_test)
end


# To see how the accuracy on the testing set keeps increasing as the training progresses.
using Plots
plot(acc_train, xlabel="Iteration", ylabel="Accuracy", label="train", ylim=(-0.01,1.01))
plot!(acc_test, xlabel="Iteration", label="test", ylim=(-0.01,1.01))