using SymEngine

"""
    kernel_coder(symbolic_kernel_original, kernel_name; periodic_var="", cutoff_var="", loc="src/kernels/")

Creates the necessary differentiated versions of base kernels required by GLOM
and saves the script containing them to `loc`

# Arguments
- `symbolic_kernel_original::Basic`: Kernel function created using variables declared with `SymEngine`'s `@vars` macro
- `kernel_name::String`: The name the kernel function will be saved with
- `periodic_var::String=""`: If changed, tries to convert the named variable (currently only one) into a periodic variable by replacing it with `2*sin(π*δ/periodic_var)`
- `cutoff_var::String=""`: If changed, makes the kernel return 0 for `abs(δ) > cutoff_var`
- `loc::String="src/kernels/"`: The location where the script will be saved

# Extra Info
The created function will look like this

    \$kernel_name(hyperparameters::Vector{<:Real}, δ::Real; dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

For example, you could define a kernel like so:

    "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
    function se_kernel_base(λ::Number, δ::Number)
        return exp(-δ ^ 2 / (2 * λ ^ 2))
    end

And then calculate the necessary derivative versions like so:

    @vars δ λ
    kernel_coder(se_kernel_base(λ, δ), "se")

The function is saved in `loc` * `kernel_name` * "_kernel.jl", so you can use it with a command akin to this:

    include(loc * kernel_name * "_kernel.jl")

See also: [`include_kernel`](@ref)

"""
function kernel_coder(
    symbolic_kernel_original::Basic,
    kernel_name::String;
    periodic_var::String="",
    cutoff_var::String="",
    loc::String="src/kernels/")
    # add_white_noise::Bool=false)

    δ = symbols("δ")
    periodic = periodic_var != ""
    if periodic
        P_sym_str = kernel_name * "_P"
        P_sym = symbols(P_sym_str)
        π_sym = symbols("π_sym")
        @assert occursin(periodic_var, SymEngine.toString(symbolic_kernel_original)) "can't find periodic variable"
        symbolic_kernel_original = subs(symbolic_kernel_original, abs(symbols(periodic_var))=>2*sin(π_sym*abs(δ)/P_sym))
        symbolic_kernel_original = subs(symbolic_kernel_original, symbols(periodic_var)=>2*sin(π_sym*δ/P_sym))
        # length(periodic_vars)==1 ? P_sym_str = kernel_name * "_P" : P_sym_str = kernel_name * "_P$i"
        # for i in 1:length(periodic_vars)
        #     length(periodic_vars)==1 ? P_sym_str = kernel_name * "_P" : P_sym_str = kernel_name * "_P$i"
        #     P_sym = symbols(P_sym_str)
        #     append!(symbols, [P_sym])
        #     append!(symbols_str, [P_sym_str])
        #     symbolic_kernel_original = subs(symbolic_kernel_original, periodic_vars[i]=>2*sin(π*δ/P_sym))
        # end
        # kernel_name *= "_periodic"
    end

    # if add_white_noise
    #     σ_white = symbols("σ_white")
    #     symbolic_kernel_original += σ_white * σ_white
    # end

    # get the symbols of the passed function
    symbs = free_symbols(symbolic_kernel_original)

    δ_inds = findall(x -> x==δ, symbs)
    @assert length(δ_inds) <= 1
    if length(δ_inds)==1
        deleteat!(symbs, δ_inds[1])
    end
    π_inds = findall(x -> x==symbols("π_sym"), symbs)
    if length(π_inds) > 0
        deleteat!(symbs, π_inds[1])
    end
    symbs_str = [string(symb) for symb in symbs]

    hyper_amount = length(symbs)

    # open the file we will write to
    kernel_name *= "_kernel"
    file_loc = loc * kernel_name * ".jl"
    io = open(file_loc, "w")

    # begin to write the function including assertions that the amount of hyperparameters are correct
    write(io, """import GPLinearODEMaker.powers_of_negative_one\n\n\"\"\"
    $kernel_name(hyperparameters, δ, dorder; shift_ind=0)

Created by kernel_coder(). Requires $hyper_amount hyperparameters.
Likely created using $(kernel_name)_base() as an input.
Use with include(\"src/kernels/$kernel_name.jl\").

# Arguments
- `hyperparameters::Vector`: The hyperparameter values. For this kernel, they should be `$symbs_str`
- `δ::Real`: The difference between the inputs (e.g. `t1 - t2`)
- `dorder::Vector{<:Integer}`: How many times to differentiate with respect to the inputs and the `hyperparameters` (e.g. `dorder=[0, 1, 0, 2]` would correspond to differentiating once w.r.t the second input and twice w.r.t `hyperparameters[2]`)
- `shift_ind::Integer=0`: If changed, the index of which hyperparameter is the `δ` shifting one
\"\"\"
function $kernel_name(
    hyperparameters::AbstractVector{<:Real},
    δ::Real,
    dorder::AbstractVector{<:Integer})

    @assert length(hyperparameters)==$hyper_amount \"hyperparameters is the wrong length\"
    dorder_len = $(hyper_amount + 2)
    @assert length(dorder)==dorder_len \"dorder is the wrong length\"
    @assert maximum(dorder) < 5 \"No more than two derivatives for each time or hyperparameter can be calculated\"
    @assert minimum(dorder) >= 0 \"No integrals\"

    dorder2 = dorder[2]
    dorder[2] += dorder[1]

    dorder_view = view(dorder, 2:dorder_len)
    \n""")

    # map the hyperparameters that will be passed to this function to the symbol names
    for i in 1:(hyper_amount)
        write(io, "    " * symbs_str[i] * " = hyperparameters[$i]\n")
    end
    # if add_white_noise
    #     write(io, "    σ_white *= Int(δ == 0)\n")  # accounting for white noise only being on the diagonal
    # end
    write(io, "\n")

    if cutoff_var!=""
        @assert cutoff_var in symbs_str
        write(io, "    if abs(δ) > $cutoff_var\n")
        write(io, "        dorder[2] = dorder2\n")
        write(io, "        return 0\n")
        write(io, "    end\n\n")
    end

    abs_vars = String[]
    for i in append!(["δ"], symbs_str)
        # println("abs($i)")
        if occursin("abs($i)", SymEngine.toString(symbolic_kernel_original))
            append!(abs_vars, [i])
            # println("found something")
            # write(io, "    dabsδ = sign(δ)  # store derivative of abs()\n")
            write(io, "    dabs_$i = powers_of_negative_one($i < 0)  # store derivative of abs()\n")
            write(io, "    abs_$i = abs($i)\n\n")
        end
    end

    max_δ_derivs = 7  # δf has 7 derivatives (0-(4+2), 4 from 2nd derivatives of GP and 2 from a possible time delay hyperparameter)
    max_hyper_derivs = 3  # other symbols have 3 (0-2)
    # calculate all of the necessary derivations we need for the GLOM model
    # for two symbols (δ and a hyperparameter), dorders is of the form:
    # [6 2; 4 2; 3 2; 2 2; 1 2; 0 2; ... ]
    # where the dorders[n, :]==[dorder of δ, dorder of symbol 2]
    # can be made for any number of symbols

    dorders = zeros(Int64, max_δ_derivs * (max_hyper_derivs ^ hyper_amount), hyper_amount + 1)
    amount_of_dorders = size(dorders,1)
    for i in 1:amount_of_dorders
          quant = amount_of_dorders - i
          dorders[i, 1] = rem(quant, max_δ_derivs)
          quant = div(quant, max_δ_derivs)
          for j in 1:(hyper_amount)
                dorders[i, j+1] = rem(quant, max_hyper_derivs)
                quant = div(quant, max_hyper_derivs)
          end
    end

    # for each differentiated version of the kernel
    for i in 1:amount_of_dorders

        # dorder = convert(Vector{Int64}, dorders[i, :])
        dorder = dorders[i, :]

        # only record another differentiated version of the function if we will actually use it
        # i.e. no instances where differentiations of multiple, non-time symbols are asked for
        if sum(dorder[2:end]) < max_hyper_derivs
            symbolic_kernel = copy(symbolic_kernel_original)

            # performing the differentiations
            symbolic_kernel = diff(symbolic_kernel, δ, dorder[1])
            for j in 1:hyper_amount
                # println(symbs[j], dorder[j+1])
                symbolic_kernel = diff(symbolic_kernel, symbs[j], dorder[j+1])
            end

            for j in abs_vars
                variable, abs_var, dabs_var = symbols("$j abs_$j dabs_$j")
                symbolic_kernel = subs(symbolic_kernel, diff(abs(variable), variable)=>dabs_var)
                for i in 2:6
                    symbolic_kernel = subs(symbolic_kernel, diff(abs(variable), variable, i)=>0)
                end
                # can't do the following:
                # for i in 1:2
                #     symbolic_kernel = subs(symbolic_kernel, dabs_var^(i*2)=>1)
                # end
                # because it solves i.e. dabs_var^(i*2)=>1 is seen as dabs_var=1
                symbolic_kernel = subs(symbolic_kernel, abs(variable)=>abs_var)
            end

            symbolic_kernel_str = SymEngine.toString(symbolic_kernel)
            symbolic_kernel_str = replace(symbolic_kernel_str, "π_sym"=>"π")
            symbolic_kernel_str = " " * symbolic_kernel_str

            # some simplifications
            symbolic_kernel_str = replace(symbolic_kernel_str, "sqrt(δ^2)"=>"abs(δ)")
            symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 - "=>" -")
            symbolic_kernel_str = replace(symbolic_kernel_str, "(0.0 - "=>"(-")
            symbolic_kernel_str = replace(symbolic_kernel_str, "(0.0 + "=>"(")
            symbolic_kernel_str = replace(symbolic_kernel_str, " 0.0 + "=>" ")
            symbolic_kernel_str = replace(symbolic_kernel_str, " 1.0*"=>" ")
            symbolic_kernel_str = replace(symbolic_kernel_str, "-1.0*"=>"-")

            # println(symbolic_kernel_str)

            write(io, "    if dorder_view==" * string(dorder) * "\n")
            write(io, "        func =" * symbolic_kernel_str * "\n    end\n\n")
        end

    end

    write(io, "    dorder[2] = dorder2  # resetting dorder[2]\n")
    write(io, "    return powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives")
    write(io, "\n\nend\n\n\n")
    write(io, "return $kernel_name, $hyper_amount  # the function handle and the number of kernel hyperparameters\n")
    close(io)

    @warn "The order of hyperparameters in the function may be different from the order given when you made them in @vars. Check the created function file (or the print statement below) to see what order you need to actually use."
    println("$kernel_name() created at $file_loc")
    println("hyperparameters == ", symbs_str)
end


"""
    include_lag_kernel(kernel_name)

Provides a kernel for modelling two variables with the same shared latent GP,
    but with a lag term between them using one of GLOM's base kernels. The
    returned kernel function should be called with the lag hyperparameter
    appended to the end of the hyperparameters for the base kernel.

# Arguments
- `kernel_name::String`: The base kernel function to use. Passed to include_kernel(kernel_name)
"""
function include_lag_kernel(kernel_name::String)

    base_kernel, base_n_hyper = include_kernel(kernel_name)

    function lag_kernel(
        hyperparameters::AbstractVector{<:Real},
        δ::Real,
        dorder::AbstractVector{<:Integer};
        outputs::AbstractVector{<:Integer}=[1,1])

        @assert length(hyperparameters)==base_n_hyper+1 "hyperparameters is the wrong length"
        @assert length(dorder)== base_n_hyper+3 "dorder is the wrong length"
        @assert maximum(dorder) < 3 "No more than two derivatives for each time or hyperparameter can be calculated"
        @assert minimum(dorder) >= 0 "No integrals"

        @assert maximum(outputs) < 3
        @assert minimum(outputs) > 0
        @assert length(outputs) == 2

        θ = view(hyperparameters, 1:length(hyperparameters)-1)
        lag = hyperparameters[end]
        # dlag = dorder[end]

        variance = outputs[1]==outputs[2]

        if dorder[end]>0 && variance; return 0 end

        dorder_no_lag = view(dorder, 1:base_n_hyper+2)

        if variance; return base_kernel(θ, δ, dorder_no_lag) end

        if outputs==[2,1]
            dorder2 = dorder_no_lag[2]
            dorder_no_lag[2] += dorder[end]
            result = base_kernel(θ, δ - lag, dorder_no_lag)
            dorder_no_lag[2] = dorder2
            return result
        end

        if outputs==[1,2]
            dorder2 = dorder_no_lag[2]
            dorder_no_lag[2] += dorder[end]
            result = powers_of_negative_one(dorder[end]) * base_kernel(θ, δ + lag, dorder_no_lag)
            dorder_no_lag[2] = dorder2
            return result
        end

        @error "You should never reach this. Some edge case is not being covered in lag_kernel()"

    end

    return lag_kernel, base_n_hyper+1  # the function handle and the number of kernel hyperparameters
end
