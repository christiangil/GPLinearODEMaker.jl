using SymEngine

"""
Creates the necessary differentiated versions of base kernels required by the GLOM et al. 2017 paper (https://arxiv.org/abs/1711.01318) method.
You must pass it a SymEngine Basic object with the variables already declared with the @vars command. δ or abs_δ must be the first declared variables.
The created function will look like this

    \$kernel_name(hyperparameters::Vector{<:Real}, δ::Real; dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

For example, you could define a kernel like so:

    "Radial basis function GP kernel (aka squared exonential, ~gaussian)"
    function rbf_kernel_base(λ::Number, δ::Number)
        return exp(-δ ^ 2 / (2 * λ ^ 2))
    end

And then calculate the necessary derivative versions like so:

    @vars δ λ
    kernel_coder(rbf_kernel_base(λ, δ), "rbf_kernel")

The function is saved in src/kernels/\$kernel_name.jl, so you can use it with a command akin to this:

    include("src/kernels/" * kernel_name * ".jl")

"""

function kernel_coder(
    symbolic_kernel_original::Basic,
    kernel_name::String;
    periodic_var::String="",
    cutoff_var::String="")

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
    # get the symbols of the passed function
    symbs = free_symbols(symbolic_kernel_original)

    δ_inds = findall(x -> x==δ, symbs)
    @assert length(δ_inds) == 1
    deleteat!(symbs, δ_inds[1])
    π_inds = findall(x -> x==symbols("π_sym"), symbs)
    if length(π_inds) > 0
        deleteat!(symbs, π_inds[1])
    end
    symbs_str = [string(symb) for symb in symbs]

    hyper_amount = length(symbs)

    # open the file we will write to
    kernel_name *= "_kernel"
    file_loc = "src/kernels/" * kernel_name * ".jl"
    io = open(file_loc, "w")

    # begin to write the function including assertions that the amount of hyperparameters are correct
    write(io, """import GPLinearODEMaker.powers_of_negative_one\n\n\"\"\"
$kernel_name function created by kernel_coder(). Requires $hyper_amount hyperparameters. Likely created using $(kernel_name)_base() as an input.
Use with include(\"src/kernels/$kernel_name.jl\").
hyperparameters == $symbs_str
\"\"\"
function $kernel_name(
    hyperparameters::Vector{<:Real},
    δ::Real;
    dorder::Vector{<:Integer}=zeros(Int64, length(hyperparameters) + 2))

    @assert length(hyperparameters)==$hyper_amount \"hyperparameters is the wrong length\"
    dorder_len = $(hyper_amount + 2)
    @assert length(dorder)==dorder_len \"dorder is the wrong length\"
    dorder2 = dorder[2]
    @assert maximum(dorder) < 3 \"No more than two time derivatives for either t1 or t2 can be calculated\"

    dorder[2] = sum(dorder[1:2])\n\n""")

    # map the hyperparameters that will be passed to this function to the symbol names
    for i in 1:(hyper_amount)
        write(io, "    " * symbs_str[i] * " = hyperparameters[$i]" * "\n")
    end
    write(io, "\n")

    if cutoff_var!=""
        @assert cutoff_var in symbs_str
        write(io, "    if abs(δ) > $cutoff_var\n")
        write(io, "        return 0\n")
        write(io, "    end\n\n")
    end

    abs_vars = String[]
    for i in append!(["δ"], symbs_str)
        println("abs($i)")
        if occursin("abs($i)", SymEngine.toString(symbolic_kernel_original))
            append!(abs_vars, [i])
            println("found something")
            # write(io, "    dabsδ = sign(δ)  # store derivative of abs()\n")
            write(io, "    dabs_$i = powers_of_negative_one($i < 0)  # store derivative of abs()\n")
            write(io, "    abs_$i = abs($i)\n\n")
        end
    end

    # δf has 5 derivatives (0-4) and the other symbols have 3 (0-2)
    max_δ_derivs = 5  # (0-4)
    max_hyper_derivs = 3  # (0-2)
    # calculate all of the necessary derivations we need for the GLOM model
    # for two symbols (δ and a hyperparameter), dorders is of the form:
    # [4 2; 3 2; 2 2; 1 2; 0 2; ... ]
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
                for i in 1:2
                    symbolic_kernel = subs(symbolic_kernel, dabs_var^(i*2)=>1)
                end
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

            write(io, "    if view(dorder, 2:dorder_len)==" * string(dorder) * "\n")
            write(io, "        func =" * symbolic_kernel_str * "\n    end\n\n")
        end

    end

    write(io, "    dorder[2] = dorder2  # resetting dorder[2]\n")
    write(io, "    return  powers_of_negative_one(dorder2) * float(func)  # correcting for amount of t2 derivatives\n\n")
    write(io, "end\n\n\n")
    write(io, "return $kernel_name, $hyper_amount  # the function handle and the number of kernel hyperparameters\n")
    close(io)

    @warn "The order of hyperparameters in the function may be different from the order given when you made them in @vars. Check the created function file (or the print statement below) to see what order you need to actually use."
    println("$kernel_name() created at $file_loc")
    println("hyperparameters == ", symbs_str)
end
