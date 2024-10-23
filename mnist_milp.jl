using JuMP
using MLDatasets
using LinearAlgebra

X_train, y_train = MNIST(split=:train)[:]
X_test, y_test = MNIST(split=:test)[:]
# X_test, y_test = MNIST.testdata() # MNIST.testdata() is deprecated, use `MNIST(split=:test)[:]` instead

X_train = reshape(X_train, 784, :)'
X_test = reshape(X_test, 784, :)'
