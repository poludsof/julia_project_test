include("first_nn.jl")
using Flux

n_hidden = 5
m = Chain(
    Dense(size(X_train,1), n_hidden, relu),
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
L(x,y) = crossentropy(m(x), y)
L(X_train, y_train) # => 1.5389597f0

# The function gradient takes two inputs.
# The first one is the function we want to differentiate, and the second one are the parameters.
grad_val = gradient(() -> L(X_train, y_train), params(X_train))
size(grad_val[X_train])