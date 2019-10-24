const CuOrAdj = Union{CuVecOrMat, LinearAlgebra.Adjoint{T, CuVecOrMat{T}}, LinearAlgebra.Transpose{T, CuVecOrMat{T}}} where {T<:AbstractFloat}

function Base.:\(_A::AT1, _B::AT2) where {AT1<:CuOrAdj, AT2<:CuOrAdj}
    A, B = copy(_A), copy(_B)
    A, ipiv = CuArrays.CUSOLVER.getrf!(A)
    return CuArrays.CUSOLVER.getrs!('N', A, ipiv, B)
end
