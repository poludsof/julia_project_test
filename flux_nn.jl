include("first_nn.jl")
using Flux

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
L(x,y) = crossentropy(m(x), y) # define loss function
L(X_train, y_train) # => 1.5389597f0

# The function gradient takes two inputs.
# The first one is the function we want to differentiate, and the second one are the parameters.
grad_val = gradient(() -> L(X_train, y_train), params(X_train))
size(grad_val[X_train])


# Train the classifiers for 250 iterations
opt = Descent(0.1)
max_iter = 250

acc_test = zeros(max_iter)
for i in 1:max_iter
    gs = gradient(() -> L(X_train, y_train), ps)
    Flux.Optimise.update!(opt, ps, gs)
    acc_test[i] = accuracy(X_test, y_test)
end


# To see how the accuracy on the testing set keeps increasing as the training progresses.
using Plots
plot(acc_test, xlabel="Iteration", ylabel="Test accuracy", label="", ylim=(-0.01,1.01))