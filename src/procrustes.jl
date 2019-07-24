struct Transformation
    R::Matrix{Float64} # Rotation
    s::Float64         # Scaling
    b::Matrix{Float64} # Bias
end

"""
procrustes(X, Y [, reflection::Union{Bool,Symbol}])

Determines a linear transformation (translation, reflection, orthogonal 
rotation, and scaling) of the points in the matrix Y to best conform 
them to the points in the matrix X. The "goodness-of-fit" criterion is
the sum of squared errors.

X and Y are assumed to have the same number of points (rows), and
the i'th point in Y to the i'th point in X is matched.  Points
in Y can have smaller dimension (number of columns) than those in X.
In this case, columns are added columns of zeros to Y as necessary.

Z, transform = procrustes(X, Y) returns the transformed Y values and
the transformation, with the fields:
 b:  the translation (bias) component
 R:  the orthogonal rotation and reflection component
 s:  the scaling component
That is,

    Z = transform.s * Y * transform.R + transform.b.

Z, transform = procrustes(X, Y; reflection=false) computes a procrustes solution
that does not include a reflection component, that is, det(transform.R) is
1.  procrustesX, Y; reflection=:best) computes the best fit procrustes
solution, which may or may not include a reflection component, :best is
the default.  procrustes(X, Y, reflection=true) forces the solution to
include a reflection component, that is, det(transform.R) is -1.

Examples:

 # Create some random points in two dimensions
 X = randn(10,2)

 # Those same points, rotated, scaled, translated, plus some noise
 S = [0.5 -sqrt(3)/2; sqrt(3)/2 0.5] % rotate 60 degrees
 Y = 0.5*X*S .+ 2 .+ 0.05*randn(10,2)

 # Conform Y to X, plot original X and Y, and transformed Y
 Z, transform = procrustes(X, Y)
 scatter(X[:,1], X[:,2])
 scatter!(Y[:,1], Y[:,2])
 scatter!(Z[:,1, Z[:,2])

References:
[1] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
[2] Gower, J.C. and Dijskterhuis, G.B., Procrustes Problems, Oxford
    Statistical Science Series, Vol 30. Oxford University Press, 2004.
[3] Bulfinch, T., The Age of Fable; or, Stories of Gods and Heroes,
    Sanborn, Carter, and Bazin, Boston, 1855.

Adapted from MATLAB's implementation of Procrustes.
"""
function procrustes(X, Y; reflection::Union{Bool,Symbol}=:best)

    n, m   = size(X)
    ny, my = size(Y)

    @assert ny == n "Input size mismatch. Number of rows has to be the same on both matrices."
    @assert my ≤ m "Y has too many columns."

    # Center at the origin.
    muX = mean(X, dims=1)
    muY = mean(Y, dims=1)
    X0 = X - repeat(muX, n, 1)
    Y0 = Y - repeat(muY, n, 1)

    ssqX = sum(X0.^2, dims=1)
    ssqY = sum(Y0.^2, dims=1)
    constX = all(ssqX .≤ abs.(eps(eltype(X))*n*muX).^2)
    constY = all(ssqY .≤ abs.(eps(eltype(X))*n*muY).^2)
    ssqX = sum(ssqX)
    ssqY = sum(ssqY)

    if !constX && !constY
        # The "centered" Frobenius norm.
        normX = sqrt(ssqX) # == sqrt(trace(X0*X0'))
        normY = sqrt(ssqY) # == sqrt(trace(Y0*Y0'))

        # Scale to equal (unit) norm.
        X0 = X0 / normX
        Y0 = Y0 / normY

        # Make sure they're in the same dimension space.
        if my < m
            Y0 = [Y0 zeros(n, m-my)]
        end

        # The optimum rotation matrix of Y.
        A = X0' * Y0
        L, D, M = svd(A)
        R = M * L'
        if reflection == :best # :best
            # Let the data decide if a reflection is needed.
        else
            have_reflection = (det(R) < 0)
            # If we don't have what was asked for ...
            if reflection != have_reflection
                # ... then either force a reflection, or undo one.
                M[:,end] = -M[:,end]
                D[end,end] = -D[end,end]
                R = M * L'
            end
        end
        
        # The minimized unstandardized distance D[X0, s*Y0*R] is
        # ||X0||^2 + s^2*||Y0||^2 - 2*s*trace(R*X0'*Y0)
        traceTA = sum(D) # == trace(sqrtm(A'*A)) when doReflection is :best
        
        # The optimum scaling of Y.
        s = traceTA * normX / normY
        
        # The standardized distance between X and s*Y*R+b.
        d = 1 - traceTA.^2

        Z = normX*traceTA * Y0 * R + repeat(muX, n, 1)
        
        if my < m
            R = R[1:my,:]
        end
        b = muX - s*muY*R
        transform = Transformation(R, s, b)

    # The degenerate cases: X all the same, and Y all the same.
    elseif constX
        d = 0
        Z = repeat(muX, n, 1)
        R = Matrix{Float64}(I, my, m)
        transform = Transformation(R, 0, muX)
    else # ~constX & constY
        d = 1
        Z = repeat(muX, n, 1)
        R = Matrix{Float64}(I, my, m)
        transform = Tranformation(R, 0, muX)
    end

    return Z, transform

end

function applyprocrustes(tr::Transformation, Y::Matrix{T}) where T <: Real
    return tr.s .* Y * tr.R .+ repeat(tr.b, size(Y, 1), 1)
end
