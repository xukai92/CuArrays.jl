# ForwardDiff integration

for f in libdevice
  if haskey(ForwardDiff.DiffRules.DEFINED_DIFFRULES, (:Base,f,1))
    f == :tanh && continue
    diffrule = ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:Base,f,1)]
    ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative,f,1)] =
      (args...) -> replace_device(diffrule(args...))
    eval(ForwardDiff.unary_dual_definition(:CUDAnative, f))
  end
end

ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :tanh, 1)] = x ->
  replace_device(:(1-tanh(x)^2))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

# Gamma functions

ForwardDiff.DiffRules.@define_diffrule CuArrays.lgamma(a) = :(CuArrays.digamma($a))
eval(ForwardDiff.unary_dual_definition(:CuArrays, :lgamma))

ForwardDiff.DiffRules.@define_diffrule CuArrays.digamma(a) = :(CuArrays.trigamma($a))
eval(ForwardDiff.unary_dual_definition(:CuArrays, :digamma))

ForwardDiff.DiffRules.@define_diffrule CuArrays.lbeta(a, b) = :(CuArrays.digamma($a) - CuArrays.digamma($a + $b)), :(CuArrays.digamma($b) - CuArrays.digamma($a + $b))
eval(ForwardDiff.binary_dual_definition(:CuArrays, :lbeta))
