using Flux
using MLDatasets
using JuMP
using GLPK
using Plots
using HiGHS

# Загрузка данных MNIST
train_x, train_y = MLDatasets.MNIST.traindata()
X_train = Float32.(reshape(train_x, 28 * 28, size(train_x, 3))) ./ 255.0  # Нормализация
y_train = Flux.onehotbatch(train_y, 0:9)  # Одноразрядное кодирование

# Определение модели
model = Chain(
    Dense(28 * 28, 128, relu),
    Dense(128, 10, identity),
    softmax
)

# Функция потерь
loss(x, y) = crossentropy(model(x), y)

# Обучение модели
opt = Flux.Descent(0.5)
for epoch in 1:10
    for i in 1:100
        x = X_train[:, i:i]  # Берем один пример
        y = y_train[:, i:i]  # Соответствующая метка
        Flux.train!(loss, params(model), [(x, y)], opt)
    end
    println("Epoch: $epoch, Loss: $(loss(X_train, y_train))")
end

# Функция для нахождения адверсариальных примеров
function adversarial(model::Chain, input::AbstractVector{<:Integer}, output, fix_inputs=Int[]; optimizer = HiGHS.Optimizer, objective = :satisfy, paranoid = false, kwargs...)
	mathopt_model = Model(optimizer)
	ivars = @variable(mathopt_model, [1:length(input)], Bin)
	ovars = setup_adversarial!(mathopt_model, model[1], ivars; kwargs...)
	!paranoid && set_silent(mathopt_model)
	all_vars = [ivars, ovars]
	for layer in model.layers[2:end]
		ovars = setup_adversarial!(mathopt_model, layer, ovars;kwargs...)
		push!(all_vars, ovars)
	end

	setup_output_constraints!(mathopt_model, ovars, output)
	for i in fix_inputs
		v = input[i] > 0
		@constraint(mathopt_model, ivars[i] == v)
	end

	set_adversarial_objective!(mathopt_model, input, ivars, objective)
	optimize!(mathopt_model)
	status = JuMP.termination_status(mathopt_model)
	# write_to_file(mathopt_model, "/tmp/model.lp")
	status == JuMP.NO_SOLUTION && return(:infeasible, input)
	status == JuMP.INFEASIBLE && return(:infeasible, input)
	x = value.(ivars)
	y = value.(ovars)
	x = [xᵢ > 0.5 ? 1 : -1 for xᵢ in x]
	(:success, x)
end

function adversarial(layer::Dense, input, output, fix_inputs=Int[]; kwargs...)
	# this is kind of bizaare, but adversarial on chain do all the heavy work.
	adversarial(Chain((layer,)), input, output, fix_inputs; kwargs...)
end

function set_adversarial_objective!(mathopt_model, input::AbstractVector{<:Integer}, ivars, objective)
if objective == :satisfy
@objective(mathopt_model, Min, 0)
elseif objective == :minimal
w = [v > 0 ? -1 : 1 for v in input]
@objective(mathopt_model, Min, dot(w, ivars))
else
error("unknown objective option, $(objective), allowed: satisfy, minimal")
end
end

"""
setup_output_constraints!(mathopt_model, ovars, output::Integer)

set the output constraints. If `output` is integer, it is assumed to be true and all
other are considered false
"""
function setup_output_constraints!(mathopt_model, ovars, output::Union{<:BitVector, <:Vector{Bool}})
(length(ovars) != length(output)) && error("The length of `output` $(length(output)) has to match length of output of model $(length(ovars)).")
@constraint(mathopt_model, dot(output, 1 .- ovars) + dot(1 .- output, ovars) ≥ 1)
end

function setup_output_constraints!(mathopt_model, ovars, output)
setup_output_constraints!(mathopt_model, ovars, Vector(output) .> 0)
end

function setup_output_constraints!(mathopt_model, ovars, output::Integer)
(output > length(ovars)) && error("output dimension of model and the output does not match")
o = [i == output for i in eachindex(ovars)]
@constraint(mathopt_model, dot(o,1 .- ovars) + dot(1 .- o, ovars) ≥ 1)
end

function setup_adversarial!(mathopt_model, layer::Dense{N,M,I}, ivars; digits = 0, kwargs...) where {N,M,I}
D = input_dim(layer)
odim = length(layer.rules)
ovars = @variable(mathopt_model, [1:odim], Bin)

δ = 10.0^-digits
for (i, r) in enumerate(layer.rules)
ii = find_ones(r.mask, D)
wᵢ = Vector(PackedVector{N,I}(r.p, D))[ii] .> 0
xᵢ = ivars[ii]
mᵢ = r.m
ub = length(ii)
@constraint(mathopt_model, dot(xᵢ, wᵢ) + dot(1 .- xᵢ, 1 .- wᵢ)  ≥ ovars[i] * mᵢ)
@constraint(mathopt_model, dot(xᵢ, wᵢ) + dot(1 .- xᵢ, 1 .- wᵢ)  ≤ (1 - ovars[i]) * (mᵢ - 1) + ovars[i]*ub)
end
ovars
end

# Остальные функции setup_output_constraints! и setup_adversarial! остаются без изменений

# Параметры для нахождения адверсариальных примеров
fixed_inputs = Int[]  # Можно указать фиксированные индексы, если необходимо

# Выбор одного изображения для тестирования
input_image = X_train[:, 1]  # Первое изображение
output_label = argmax(y_train[:, 1])  # Соответствующий выход

# Поиск адверсариального примера
int_vector_rounded = map(x -> round(Int, x), input_image)
status, adversarial_input = adversarial(model, int_vector_rounded, output_label, fixed_inputs)

if status == :success
    println("Adversarial input found.")
    # Визуализируем изображения
    plot_images(input_image, adversarial_input)
else
    println("No adversarial input found.")
end

# Функция для отображения изображений
function plot_images(original, adversarial)
    # Изменяем размер изображений на 28x28
    original_image = reshape(original, 28, 28)
    adversarial_image = reshape(adversarial, 28, 28)

    # Создаем подграфики для отображения
    p = plot(layout=(1, 2), size=(800, 400))
    # Отображаем оригинальное изображение
    plot!(p[1, 1], original_image, color=:gray, title="Original Image", legend=false)
    # Отображаем адверсариальное изображение
    plot!(p[1, 2], adversarial_image, color=:gray, title="Adversarial Image", legend=false)
    
    return p
end
