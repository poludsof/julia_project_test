"""
adversarial(r, model::Chain, input, output; fixed_input=LiteralRule[], objective = :satisfy, optimizer = :highs, paranoid = false, kwargs...)

finds an adversarial sample `x` such that `model(x) ≠ output`. The `output` can be specified
as a binary vector or alternatively it can be a `Vector{<:LiteralRule}` in which case `rule(output)(model(x))` has
to evaluate to false (in other words the `model(x)` has to differ at least in one item specified in output).
If `fixed_input` is contains some values, the found adversarial input `x` is restricted tp contain those.
If `objective = :satisfy`, then any adversarial sample is sufficient, if `objective = :minimal`, `x` should have
hamming minimal distance to `input`.
If `paranoid` is true, then additional check of correctness are invoked.
`optimizer` allows to select the optimizer. The default is `HiGHS.Optimizer` and we test `ConstraintSolver.Optimizer`
"""
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

function adversarial(layer::PackedDense, input, output, fix_inputs=Int[]; kwargs...)
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

function setup_adversarial!(mathopt_model, layer::PackedDense{N,M,I}, ivars; digits = 0, kwargs...) where {N,M,I}
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
