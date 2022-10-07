% -- Function File: function [m, chi2r, C, iter, a, Wm] = gls_mean(X, W=[], M=[], P=[], m0=[], ival=[], opts=[])
% -- Function File: function opts = gls_mean("options")
%
%  Compute the generalized least squares mean of a set of N observation vectors 
% of size Nx that are contained in X = [ X(1), ..., X(N) ].
%   
%  The generalized least squares mean is defined as the vector m of size Nm that 
% minimizes the reduced chi-square defined by
%
%             N
%           -----
%           \                               2
%   chi2r = /     | W(i) * [ X(i) - Y(i) ] |  / dof                          (1)
%           -----
%           i = 1
%
% where dof is the degree of freedom of the problem. Each observation is 
% modelled as a linear combination of m and of a user-defined matrix P of size 
% ( Nm * Np ) with respective linear coefficients s(i) and f(i):
%
%   Y(i) = M(i) * [ P * f(i) + m * s(i) ].                                   (2)
%
% M(i) is a transformation matrix of size ( Nx * Nm ) which projects m and P in 
% the space of X(i). W(i) is a weight matrix of size ( Nw * Nx ) that is 
% associated with the observation X(i). W(i) is the pseudo-inverse of 
% symmetrical decomposition, L(i), of the covariance matrix associated with 
% X(i), Cx(i) = L(i) * L(i)', where L(i) has a size of ( Nx * Nw ) with 
% Nw <= Nx. If Nx=Nw, then W(i) can be computed as the inverse of the Cholesky 
% decomposition of Cx(i). In Equation 2, m, s(i) and f(i) are free to 
% vary. However, we can compute the values of s(i) and f(i) that minimize 
% Equation 1 for a given value of m such that it actually only depends on m.
%
%   The input set of observations, X, is split by this function into a training 
% set of observations, whose indices are not contained in ival, and a validation 
% set of observations whose indices are contained in ival. Equation 1 is then 
% evaluated on both the training set and validation set of observations 
% independently, although m is only inferred based on the training set of 
% observations. Convergence can be probed using the chi2r evaluated on the 
% validation set of observations using the early_stop option.
%
%   The likelihood associated with Equation 1 is maximized through an iterative 
% Expectation-Maximization algorithm, hence converging to the nearest local 
% minimum of the chi-square function. In order to mitigate the problem of local
% convergence, one can use a subset of the training set of observation at each 
% iteration of the algorithm by setting the fbatch option to a value lower than 
% one. Complementarily, one can also use momentum by setting the beta option to 
% a value different than zero. If available, an initial estimate of the 
% generalized least squares mean can also be provided through the m0 parameter.
%
% The algorithm stops when any of the following criteria are met
%   - chi2r_best - chi2r < chi2tol for more than n_no_improve iterations where
%     chi2r_best is the best chi2r encountered so far. That is: the improvement
%     in the reduced chi-square is lower than chi2tol for n_no_improve 
%     successive iterations.
%   - The relative change in the mean vector falls under a given threshold, 
%     rtol.
%   - A maximum number of iterations is reached, maxiter
%
%   The value of m associated with the minimal chi2r is then returned along with 
% the final value of this reduced chi-square, the covariance matrix on m, the 
% final number of observations that were performed, the linear coefficients 
% associated with P and m for each observation and a weight matrix on m.
%
% Input:  
% ------  
%       X: The matrix of observations, size of ( Nx * N ).
%       W: The weights matrices associated with the observations, size of 
%          ( Nw * Nx * N ), or ( Nx * N ) if columns of W are the inverse 
%          (uncorrelated) standard deviations on X. 
%          If empty, unit weights are considered.
%       M: Transformation matrices allowing to project m and P in the space of 
%          X. Size of (Nx * Nm * N) if each observation is associated with its
%          own transformation matrix or size of (Nx * Nm) in order to use a 
%          global transformation matrix.
%          If empty, M is taken as the identity matrix of size (Nx * Nx).
%       P: The set of fixed templates to use in order to fit the background 
%          signal, size of ( Nm * Np ).
%          If empty, P will not be used.
%      m0: The column vector containing an initial estimate of m, size of Nm. 
%          If empty, m0 is computed from the the observation having the highest
%          signal-to-noise ratio.
%    ival: Indices of the set of validation observations. Should be a vector 
%          containing indices in [1, N].
%    opts: Structure containing the options to pass to the algorithm.
%          This structure contains
%            maxiter: The maximum number of iteration to perform.
%            chi2tol: The tolerance in the change of the reduced chi-square 
%                     value. Reduced chi-square is evaluated on either the
%                     training set of observations or on the validation set of
%                     observations, depending on the value of the early_stop
%                     option.
%       n_no_improve: Stop if the reduced chi-square is higher than 
%                     chi2r_best - chi2tol for more than n_no_improve 
%                     iterations where chi2r_best is the best chi2r encountered 
%                     so far.
%               rtol: The tolerance in the relative change of the mean vector.
%                     The algorithm stops if rtolf(p,dp,rtol) <= rtol. See rtolf
%                     for more details.
%              rtolf: The function used to compute the tolerance in the relative 
%                     change of parameters. 
%                       rtolf = @(m, dm, rtol) where m are the current mean, dm 
%                     the last change in the mean and rtol is described above.
%             fbatch: The fraction of training observations used in batch 
%                     optimization, i.e. only a random subset of observations 
%                     will be used while fitting m in each iteration of the 
%                     Expectation-Maximization algorithm. Values in 
%                     0 < fbatch <= 1.
%               beta: Decay factor used in the computation of the momentum of m.
%                     Each update brought to m will be of the form
%                       mv(i) = beta * mv(i-1) + (1 - beta) * dm
%                       m = m + mv(i)
%                     where dm is the change in m from a least squares solution
%                     to Equation 1. Values in 0 <= beta < 1, where beta=0 
%                     implies no use of momentum.
%         early_stop: If set to 1, use the validation set of observations 
%                     instead of the training set of observation in order to 
%                     compute chi2r from Equation 1. Values in [0, 1].
%     move2best_only: Should we move to a new solution only if it is better than 
%                     the best encountered so far? If not, restart from the best
%                     solution, without momentum. Values in [0, 1].
%            verbose: Should we have to be verbose? Values in [0, 1].
%             lambda: Thikonov regularization factor of the 
%                     Expectation-minimization algorithm. Chi-square 
%                     minimizations of m in the Expectation-minimization 
%                     algorithm will be of the form
%                       chi2 = | A * m - y |^2 + lambda * | L * m |^2
%                     where L = max(sqrt(diag(A'*A))), according to 
%                     Levenberg (1944). Setting lambda to a non-zero value 
%                     provides a numerically more stable convergence at the 
%                     expense of being slower. Values in lambda >= 0.
%         lambda_cov: The Thikonov regularization factor used in order to 
%                     compute the covariance matrix in case it is close to 
%                     singular. Given the Jacobian matrix on m, J, the 
%                     covariance matrix is computed as 
%                       C = iJ * iJ' 
%                     where iJ = inv( J' * J + lambda_cov * L ) * J' and 
%                     L = diag(sqrt(J'*J)) in accordance with Marquardt (1963).
%                     Set this parameter to a large value (e.g. usually 0.001) 
%                     if you expect/notice that the problem is close to be 
%                     singular or that the errors seem unrealistic. Values in 
%                     lambda_cov >= 0.
%              lsqrf: The function to call for solving the least squares
%                     problems | y - A * x |^2  + lambda * | L * x |^2 where
%                     L is a user-defined regularization matrix and lambda is
%                     the previously defined options. lsqrf must be of the form
%                     x = lsqrf(A, y, lambda). If empty, lsqrf will be taken as
%                     an internal function based on conjugate gradient descent.
%               seed: The random seed to use for selecting batch observations.
%
% Output:
% -------
%     m: The mean vector minimizing Equation 1, size of Nm. Note that
%        m will have a norm of one and is explicitly made orthogonal to P.
% chi2r: The reduced chi-square computed on either the training set of 
%        observations or on the validation set of observations, depending on the
%        value of the early_stop option.
%     C: An asymptotic estimation of the covariance on the mean vector, size of
%        ( Nm * Nm ).
%  iter: The number of iterations that were performed.
%     a: The least squares linear coefficients associated with [P m], size of 
%        ( Np+1 * N ), I.e. a(:,i) = [f(i); s(i)].
%    Wm: The weight matrix associated with m, such that Wm'*Wm = inv(C).
%        Wm aims to be used in generalized least
%        square problems of the form | A * x - m |^2 whose solution are given by
%        x = inv( A' * inv(C) * A ) * A' * inv(C) * m
%          = inv( A' * Wm' * Wm * A ) * A' * Wm' * Wm * m.
%
% Prerequisites
% -------------
%   gls_mean run on GNU Octave 4.2.2+ <https://www.gnu.org/software/octave/>. 
% 
%   File gls_mean.m should either be in the current working directory or in the 
% Octave's function search path (see the Octave addpath function for more 
% information on how to add search paths). 
%
% Demo
% ----
%   A demonstration script can be run by typing "demo gls_mean" at the Octave
% prompt.
%
% License
% -------
%   This program is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the Free 
% Software Foundation, in version 3 of the License.
%
%   This program is distributed in the hope that it will be useful, but WITHOUT 
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License along with 
% this program. If not, see <https://www.gnu.org/licenses/>
%
% Version:
% --------
%   0.7.1
%
% History:
% --------
%   0.7.1 2022-09-29 Added a progress report while searching for the highest SNR
%                    observation, optimisation of the SNR computation in the 
%                    case of uncorrelated noise.
%   0.7.0 2022-06-01 Degree of freedom increased by 1 in order to take into 
%                    account the fact that the mean vector is normalized.
%   0.6.0 2022-03-31 Fix error in the computation of the signal-to-noise ratio
%                    Documentation update after review by C.A.L Bailer-Jones
%   0.5.0 2022-03-29 Fix minor convergence issues
%                    Add warning messages and debug facilities
%                    Exhaustive tests
%   0.4.0 2022-03-16 Add the lsqrf option
%   0.3.0 2022-03-15 Rewrite of the gls_mean_lsqrf function
%                    Removal of the lsqrSOL.m dependency
%   0.2.0 2022-03-11 Add validation set of observations
%                    Add n_no_improve, early_stop and seed options
%                    Documentation rewriting
%   0.1.0 2021-11-22 Initial functional version, used to compute the preliminary 
%                    quasar composite spectra in the first submitted version of
%                    Gaia collaboration, C.A.L Bailer-Jones et al. (2022) 
%   0.0.0 2021-08-27 Proof of concept/beta version called fitMean
%
% Bugs
% ----
%  Please report any bugs to <ldelchambre@uliege.be>.
%
% Reference
% ---------
%   Gaia collaboration et al., 2022, A&A, Gaia Data Release 3. The extragalactic 
%     content 
%   DOI: https://doi.org/10.1051/0004-6361/202243232
%   arXiv: 2206.05681
%
% Acknowledgements
% ----------------
%  C.A.L Bailer-Jones for reviewing this documentation.
function [m, chi2r, C, iter, a, Wm] = gls_mean(X, W=[], M=[], P=[], m0=[], ival=[], opts=[])
  
  # If gls_mean is called as opts = gls_mean("options"), return the default options
  if(nargin == 1 && nargout == 1 && ischar(X))
    assert(strcmpi(X, "options"), sprintf("Unrecognised input parameter \"%s\"",X));
    opts.maxiter=1024;
    opts.chi2tol=0.001;
    opts.n_no_improve=3;
    opts.rtol=1e-6;
    opts.rtolf=@(m,dm,rtol) max(abs(dm)./(abs(m)+abs(rtol)));
    opts.fbatch=1;
    opts.beta=0;
    opts.early_stop=0;
    opts.move2best_only=0;
    opts.verbose=0;
    opts.lambda=0.001;
    opts.lambda_cov=1e-14;
    opts.lsqrf = [];
    opts.seed = 1;
    m = opts;
    return;
  end
  
  # Should we plot the intermediate means for debug purposes?
  plot_mean = 0;
  
  ### Check X and get Nx, N
  sx = size(X);
  assert(numel(sx) == 2, "X has a wrong size");
  assert(all(isfinite(X(:))), "X contains non-finite values");
  Nx = sx(1);
  N = sx(2);
  
  ### Check W and get Nw
  if(isempty(W))
    W = ones(Nx, N);
  end
  sw = size(W);
  if(numel(sw) == 2)
    Nw = Nx;
    assert(all(sw == [Nw N]), "W has a wrong size");
    assert(all(sumsq(W,1) != 0), "Some observations have no valid weights in W");
  elseif(numel(sw) == 3)
    Nw = sw(1);
    assert(all(sw == [Nw Nx N]), "W has a wrong size");
    assert(all(sumsq(sumsq(W,2),1) != 0), ...
                                 "Some observations have no valid weights in W");
  else
    error("Invalid size of the weight matrix: %s", num2str(sw));
  end
  assert(all(isfinite(W(:))), "W contains non-finite values");
  
  ## Check M and get Nm
  if(isempty(M))
    M = eye(Nx);
  end
  sm = size(M);
  if(numel(sm) == 2)
    Nm = sm(2);
    assert(all(sm == [Nx Nm]), "M has a wrong size");
  elseif(numel(sm) == 3)
    Nm = sm(2);
    assert(all(sm == [Nx Nm N]), "M has a wrong size");
  else
    error("Invalid size of the transformation matrix: %s", num2str(sm));
  end
  assert(all(isfinite(M(:))), "M contains non-finite values");
  
  ### Check P and get Np
  if(isempty(P))
    P = zeros(Nm,0);
  end
  sp = size(P);
  assert(numel(sp) == 2 && all(sp == [Nm sp(2)]), "P has a wrong size");
  assert(all(isfinite(P(:))), "P contains non-finite values");
  Np = sp(2);
  
  ### Check m0
  if(~isempty(m0))
    assert(all(size(m0) == [Nm 1]), "m0 has a wrong size");
    assert(all(isfinite(m0)), "m0 contains non-finite values" );
  end
  
  ### Check ival and compute itrain, nval and ntrain
  nval = numel(ival);
  ntrain = N - nval;
  if(nval > 0)
    ival = unique(ival(:));
    assert(all(mod(ival,1)==0 & ival > 0 & ival <= N), ...
          sprintf("ival must contain integer values in the range [1, %d]", N));
    itrain = setdiff([1:N]', ival);
  else
    itrain = [1:N]';
  end
  assert(ntrain > 0, "No training set of observation is provided");
  
  ### Check options
  if(isempty(opts))
    opts = gls_mean("options");
  end
  assert(isfield(opts, "maxiter") && isscalar(opts.maxiter), "Bad maxiter option");
  assert(isfield(opts, "chi2tol") && isscalar(opts.chi2tol), "Bad chi2tol option");
  assert(isfield(opts, "n_no_improve") && isscalar(opts.n_no_improve), "Bad n_no_improve option");
  assert(isfield(opts, "rtol") && isscalar(opts.rtol), "Bad rtol option");
  assert(isfield(opts, "rtolf") && is_function_handle(opts.rtolf), "Bad rtolf option");
  assert(isfield(opts, "fbatch") && 0 < opts.fbatch && opts.fbatch <= 1, "Bad fbatch option");
  assert(isfield(opts, "beta") && isscalar(opts.beta) && 0 <= opts.beta && opts.beta < 1, "Bad beta option");
  assert(isfield(opts, "early_stop") && isscalar(opts.early_stop), "Bad early_stop option");
  assert(isfield(opts, "move2best_only") && isscalar(opts.move2best_only), "Bad move2best_only option");
  assert(isfield(opts, "verbose") && isscalar(opts.verbose), "Bad verbose option");
  assert(isfield(opts, "lambda") && isscalar(opts.lambda), "Bad lambda option");
  assert(isfield(opts, "lambda_cov") && isscalar(opts.lambda_cov), "Bad lambda_cov option");
  assert(isfield(opts, "lsqrf") && ( is_function_handle(opts.lsqrf) || isempty(opts.lsqrf) ), "Bad lsqrf option");
  assert(isfield(opts, "seed") && isscalar(opts.seed), "Bad seed option");

  # Initialize the random number generator
  rand("seed", opts.seed);
  
  # Check that ival is not empty if we should use the early stopping
  assert(opts.early_stop == 0 || numel(ival) > 0, "No validation set of observation is provided");
  
  # Initialize m0 if it is empty
  if(isempty(m0))
    # Get the training observation having the highest SNR as initial guess
    best_snr2 = -Inf; ibest = 0; tid = tic(); t = 0;
    for i=itrain'
      Xi = X(:,i);
      if(numel(size(W)) == 2)
        Wi = W(:,i);
        ig = ( Wi != 0 );
        snr2 = sumsq( Xi(ig) ) / sumsq( 1 ./ Wi(ig) ) - 1;
      else
        Wi = W(:,:,i);
        ig = ( sumsq(Wi,1) > 0 );
        di = svd(Wi(:,ig));
        snr2 = sumsq( Xi(ig) ) / sumsq( 1 ./ di(di > max(size(Wi)) * eps * di(1)) ) - 1;
      end
      if(snr2 > best_snr2)
        best_snr2 = snr2;
        ibest = i;
      end
      t = toc(tid);
      if(opts.verbose != 0 && t > 15)
        printf("\rSearching for the highest SNR observation... %.3f%% done (obs: %d, snr: %9.3e)", 100*i/ntrain, ibest, sqrt(best_snr2));
      end
    end
    if(opts.verbose != 0)
      if(t > 15) printf("\n"); end
      printf("Selecting observation %d as initial guess (snr: %g)\n", ibest, sqrt(best_snr2));
    end
    Xi = X(:,ibest);
    Wi = gls_mean_getW(W,ibest);
    Mi = gls_mean_getM(M,ibest);
    a = zeros(Np+1,N);
    a(:, ibest) = [ ( Wi * Mi * P ) \ ( Wi * Xi ); 1];
    m0 = gls_mean_dm(X, W, M, [P zeros(Nm,1)], a, ibest, opts.lambda, opts.lsqrf, opts.verbose);
  end
  
  # Compute the degrees of freedom of the problem. 
  #  - dof is the degree of freedom that would result from the fit of m to the 
  #    whole set of observations.
  #  - dof_train and dof_val are the degree of freedom that results from the 
  #    training and validation set of observations, respectively.
  if(ndims(W) == 3)
    ndata =  arrayfun( @(i) rank( W(:,:,i) ), [1:N]' );
  else
    ndata = sum( W != 0, 1 );
  end
  dof = sum(ndata) - ( Nm + N * ( Np + 1 ) - 1 );
  dof_train = sum(ndata(itrain)) - ( Nm + ntrain * ( Np + 1 ) - 1 );
  dof_val = max(sum(ndata(ival)) - ( Nm + nval * ( Np + 1 ) - 1 ), 0);
  
  # Initialize m and orthogonalize it with respect to P
  m = m0;
  m = m - P * ( P' * m );
  m = m / norm(m);
  
  # Find the least-square coefficients associated with T = [P m]
  a = gls_mean_da(X, W, M, [P m], zeros(Np+1, N), opts.lsqrf, opts.verbose);
  
  # Compute the initial reduced chi-squares
  chi2s = gls_mean_chi2(X, W, M, [P m], a);
  chi2r = sum(chi2s) / dof;
  chi2r_train = sum(chi2s(itrain)) / dof_train;
  if(dof_val > 0) chi2r_val = sum(chi2s(ival)) / dof_val; else chi2r_val = 0; end
  
  # Initialize the best reduced chi-square, mean vector and linear coefficients
  if(opts.early_stop == 0) chi2r_best=chi2r_train; else chi2r_best=chi2r_val; end 
  mbest = m;
  abest = a;
  
  # Print debug message
  if(opts.verbose != 0)
    printf("Initial reduced chi-squares: %g (on training set: %g, on validation set: %g)\n", chi2r, chi2r_train, chi2r_val);
  end
  
  # Main loop of the expectation-maximization algorithm
  iter = i_no_improve = mv = 0; dm = Inf(Nm,1);
  while(iter < opts.maxiter ...
        && i_no_improve < opts.n_no_improve ...
        && any(opts.rtol < opts.rtolf(m,dm,opts.rtol)))
    # Increase the number of iterations
    iter++;
    
    # Compute the change in the mean vector
    dm = gls_mean_dm(X, W, M, [P m], a, itrain(randperm(ntrain, ceil(opts.fbatch*ntrain))), opts.lambda, opts.lsqrf, opts.verbose);
    mv = opts.beta * mv + (1.0 - opts.beta) * dm;
    if(plot_mean != 0)
      clf; subplot(211); plot(m, "k-;Old;", m+mv, "r-;New;"); title("Mean");
    end
    m = m + mv;
    
    # Find the least-square coefficients, a
    da = gls_mean_da(X, W, M, [P m], a, opts.lsqrf, opts.verbose);
    a = a + da;
    if(plot_mean != 0)
      subplot(212); semilogy(sort(abs(a'))); title("Coefficients"); 
      legend(num2str([1:size(a,1)]', "Component %d"));
      pause(0.5);
    end
    
    # Compute the associated reduced chi-squares
    chi2s = gls_mean_chi2(X, W, M, [P m], a);
    chi2r = sum(chi2s) / dof;
    chi2r_train = sum(chi2s(itrain)) / dof_train;
    if(dof_val > 0) chi2r_val = sum(chi2s(ival)) / dof_val; else chi2r_val = 0; end
    if(opts.early_stop == 0) chi2ri=chi2r_train; else chi2ri=chi2r_val; end 
    
    # Check if the reduced chi-square improved compared to the best chi2r
    if(chi2r_best - chi2ri < opts.chi2tol)
      i_no_improve++;
    else
      i_no_improve = 0;
    end
    
    # Print debug message in verbose mode
    if(opts.verbose != 0)
      printf("%d) Reduced chi-square: %g, best=%g, diff=%g, rtol=%g (on training set: %g, on validation set: %g)\n", ...
              iter, chi2r, chi2r_best, chi2ri-chi2r_best, max(opts.rtolf(m,dm,opts.rtol)), chi2r_train, chi2r_val);
    end
    
    # Select the mean vector having the lowest chi2r
    if(chi2ri < chi2r_best)
      chi2r_best = chi2ri;
      mbest = m;
      abest = a;
    elseif(opts.move2best_only != 0)
      # If we should only move to the new solution if it is has a better chi2r
      # and this solution is worst than the previous one, keep the best one
      m = mbest;
      a  = abest;
      mv = 0; # Restart from the best estimate without momentum
    end
  end
  
  # Select the best mean vector as the one we will return
  m = mbest;
  
  # Orthogonalize m with respect to P
  m = m - P * ( P' * m );
  m = m / norm(m);
  
  # Find the final coefficients and associated chi-square
  da = gls_mean_da(X, W, M, [P m], a, opts.lsqrf, opts.verbose);
  a = a + da;
  chi2s = gls_mean_chi2(X, W, M, [P m], a);
  chi2r = sum(chi2s) / dof;
  chi2r_train = sum(chi2s(itrain)) / dof_train;
  if(dof_val > 0) chi2r_val = sum(chi2s(ival)) / dof_val; else chi2r_val = 0; end
  
  # Print the final chi-square in verbose mode
  if(opts.verbose != 0)
    printf("Final reduced chi-square: %g (on training set: %g, on validation set: %g)\n", chi2r, chi2r_train, chi2r_val);
  end
  
  # Compute the covariance matrix on the mean vector if needed
  if(isargout(3) || isargout(6))
    # Compute the iterative QR decomposition of the Jacobian matrix, J
    # such that J' * J = RP * R' * R * RP'
    R = zeros(Nm); RP = eye(Nm);
    for i=itrain'
      # Get Wi, Mi, si, Ti
      Wi = gls_mean_getW(W,i);
      Mi = gls_mean_getM(M,i);
      si = a(end,i);
      Ti = Mi * [P m];
      
      # Build the ith Jacobian matrix, Ji
      [~,Ri] = qr(Wi * Ti, 0);
      Ji = Ti * ( Ri \ ( Ri' \ ( ...
             [ ...
               -si * (Mi * P)'; ...
               ( X(:,i) - Mi * [P 2*m] * a(:,i) )' ...
             ] * Wi' * Wi * Mi ) ) ) ...
           + Mi * si;

      # Accumulate the QR decomposition of each Jacobian matrix in R
      [~, R, RP] = qr([R * RP'; Wi * Ji]);
      R = R(1:Nm,:);

			# Print progress in verbose mode
      if(opts.verbose != 0)
        printf("\rComputing the covariance matrix on m... %.3f%% done", 100*i/N);
      end
    end
    
    # Compute the weight matrix associated with m, Wm
    Wm = R * RP';
    
    # Compute the covariance matrix on m
    if(isargout(3))
      RL = [R; sqrt(opts.lambda_cov) * diag(sqrt(sumsq(R))) * eye(Nm)];
      RLc = rcond(RL'*RL);
      if(RLc < eps)
        warning(strcat( ...
                sprintf("Inverse covariance matrix on m is ill-condtioned (rcond=%g).", RLc), ...
                "The resulting covariance matrix can hence be inaccurate.", ...
                "Try increasing the lambda_cov parameter."));
      end
      iR = RL \ [eye(Nm); zeros(Nm)];
      C =  RP * ( iR * iR' ) * RP';
    end

    if(opts.verbose != 0)
      printf("\n");
    end
  end
end

# Solve the least square system A * x = y using Thikhonov regularization
function [x,cvg,iter,gnorm] = gls_mean_lsqrf(A, y, lambda)
  n = size(A,2);
  % L = sqrt(lambda * diag(sumsq(A,1))); % Marquardt regularization
  L = sqrt(lambda * max(sumsq(A,1)) * eye(n)); % Levenberg regularization
  
  # Function parameters
  precond = 0; % Should we use preconditioning?
  tol = eps * n; % The relative tolerance of the gradient
  maxiter = 128 * n; % The maximal number of iterations to perform
  nrestart = n + 1; % Restart the algorithm every nrestart iterations
  nrecompute = 1; % Explicitly recompute the residuals every nrecompute iterations
  verbose = 0; % Should we have to be verbose?
  x0 = zeros(n,1); % The initial estimate of x
  
  # Treat the special case where n == 1 apart as it is 
  # very easy and quick to solve
  if(n == 1)
    x = (A'*y) / ( A'*A + L'*L );
    cvg = 1; iter = 0; gnorm = abs( A' * ( y - A * x ) - L' * ( L * x ) );
    return;
  end
  
  # Build the pre-conditioning matrix
  if(precond != 0) 
    [~,R,p] = qr(A,0); 
    ip=zeros(1,n); 
    ip(p)=1:n; 
    M = R(:,ip); 
  else
    M = eye(n);
  end

  # Main loop of the conjugate gradient algorithm
  r = y - A * x0;
  g = A' * r - L' * ( L * x0 );
  g2 = g' * g;
  z = M \ ( M' \ g );
  gz = g' * z;
  q = z;
  etol2 = tol^2 * sumsq( A' * y );
  iter = bad_iter = 0;
  xbest = x = x0; g2best = g2;
  while(iter < maxiter && g2 > etol2)
    iter = iter + 1;
    alpha = gz / ( sumsq( A * q ) + sumsq( L * q ) );
    x = x + alpha * q;
    if(mod(iter,nrecompute) == 0)
      r = y - A * x;
      g = A' * r - L' * ( L * x );
    else
      r = r - alpha * A * q;
      g = A' * r - alpha * L' * ( L * q );
    end
    z = M \ ( M' \ g );
    g2 = g' * g; 
    if(verbose != 0) 
      printf("%d) Norm of chi2 gradient=%g (best=%g), norm of residuals=%g\n", ...
              iter, sqrt(g2), sqrt(g2best), norm(r));
    end
    if(g2 < g2best) 
      xbest = x;
      g2best = g2;
    end
    gzp = gz; gz = g' * z;
    if(mod(iter,nrestart) == 0)
      q = z;
    else
      q = z +  gz / gzp * q;
      if(sumsq(q) == 0) q = z; end
    end
  end

  # Return the best estimate
  x = xbest;
  cvg = ( g2best <= etol2 );
  gnorm = sqrt(g2best);
end

# Get the transformation matrix, Mi, that is associated with the ith observation
function Mi = gls_mean_getM(M, i)
  if(numel(size(M)) == 2)
    Mi = M;
  else
    Mi = M(:,:,i);
  end
end

# Get the weight matrix, Wi, that is associated with the ith observation
function Wi = gls_mean_getW(W, i)
  if(numel(size(W)) == 2)
    Wi = diag(W(:,i));
  else
    Wi = W(:,:,i);
  end
end

# Fit m = m0 + dm in Equation 1 given T = [ P m0 ] and a
function dm = gls_mean_dm(X, W, M, T, a, iobs, lambda, lsqrf, verbose)
  # Get the problem size
   N = numel(iobs);
  Nx = size(X, 1);
  Nm = size(M, 2);
  Nw = size(W, 1);
  
  # Process cases where the noise is uncorrelated and where M is a diagonal 
  # matrix of size (Nx x Nx) apart as it allows to significantly fasten the 
  # code execution.
  if(numel(size(W)) == 2 ...
     && numel(size(M)) == 2 ...
     && Nx == Nm && isdiag(M) ...
     && isempty(lsqrf))
    s = a(end,iobs);
    A = sumsq( M * W(:,iobs) * diag(s), 2);
    y = sum( ( ( M * W(:,iobs).^2 ) .* ( X(:,iobs) - M * T * a(:,iobs) ) * diag(s) ) , 2);
    dm = y ./ ( A * (1 + lambda) );
    dm(~isfinite(dm)) = 0;
    return;
  end
  
  # Compute the design and image matrix of the least squares problem
  A = y = []; tid = tic(); t = 0;
  for i=1:N
    iiobs = iobs(i);
    Wi = gls_mean_getW(W,iiobs);
    Mi = gls_mean_getM(M,iiobs);
    Ai = a(end,iiobs) * Wi * Mi;
    yi = Wi * ( X(:,iiobs) - Mi * T * a(:,iiobs) );
    A = [ A; Ai ]; y = [ y; yi ];
    # Use incremental QR decomposition in order to maintain A as small as
    # possible
    if(size(A,1) > size(A,2))
      [y, A] = qr(A, y);
      A = A(1:size(A,2),:);
      y = y(1:size(A,2));
    end
    # Print progress if it takes mare than 15 seconds
    t = toc(tid);
    if(verbose != 0 && t > 15)
      printf("\rUpdating the mean vector... %.3f%% done.", 100*i/N);
    end
  end
  if(verbose != 0 && t > 15)
    printf("\n");
  end
  
  # Compute dm
  ig = ( sumsq(A, 1) > 0 );
  dm = zeros(Nm,1);
  if(isempty(lsqrf))
    [dmi,cvg,iter,gnorm] = gls_mean_lsqrf(A(:,ig), y, lambda);
    # Check for convergence
    if(cvg == 0)
      warning(sprintf("The conjugate gradient descent did not converge after %d \
iterations (gradient norm = %g). Try increasing the lambda parameter.", ...
              iter, gnorm));
    end
    dm(ig) = dmi;
  else
    dm(ig) = lsqrf(A(:,ig), y, lambda);
  end
end

# Fit a = a0 + da in Equation 1 given T and a0
function da = gls_mean_da(X, W, M, T, a0, lsqrf, verbose)
  # Should we plot the fit of the template to the observation for debugging
  # purposes
  plot_fit = 0;
  
  # Get the problem size
   N = size(X, 2);
  Nx = size(X, 1);
  Nm = size(M, 2);
  Nw = size(W, 1);
  Np = size(T,2) - 1;
  
  # Process cases where only one template should be fitted to observation with 
  # uncorrelated noise and a global transformation matrix apart as it allows to
  # significantly fasten the code execution
  if(Np == 0 ...
     && numel(size(W)) == 2 ...
     && numel(size(M)) == 2 ...
     && isempty(lsqrf))
    A =  T' * M' * ( W.^2 .* repmat(M * T, 1, N) );
    y = T' * M' * ( W.^2 .* ( X - M * T * a0 ) );
    da = y ./ A;
    da(~isfinite(da)) = 0;
    return;
  end

  # Compute da
  da = zeros(Np+1, N); tid=tic(); t=0;
  for i=1:N
    Wi = gls_mean_getW(W,i);
    Mi = gls_mean_getM(M,i);
    Xi = X(:,i) - Mi * T * a0(:,i);
    ig = ( sumsq(Mi,1)' > 0 & sumsq(T,2) > 0 );
    if(sum(ig) > 0)
      A = Wi * Mi(:,ig) * T(ig,:);
      y = Wi * Xi;
      if(isempty(lsqrf))
        da(:,i) = gls_mean_lsqrf(A, y, 0);
      else
        da(:,i) = lsqrf(A, y, 0);
      end
    else
      da(:,i) = -da(:,i);
    end
    
    # Plot result for debug purposes
    if(plot_fit != 0)
      clf;plot(Mi \ X(:,i),"k-;Observation;",  ...
               T * ( a0(:,i) + da(:,i) ), "r-;Template;"); 
      title(sprintf("Observation %d (chi2: %g)", ...
                    i, sumsq(Wi*(X(:,i)-Mi*T*(a0(:,i)+da(:,i))))));
      pause(0.5);
    end
    
    # Print debug message after 15 seconds in verbose mode
    t = toc(tid);
    if(verbose != 0 && t > 15)
      printf("\rComputing linear coefficients... %.3f%% done.", 100*i/N);
    end
  end
  if(verbose != 0 && t > 15)
    printf("\n");
  end
end

# Compute the chi2 values
function chi2 = gls_mean_chi2(X, W, M, T, a)
  # Get the problem size
   N = size(X, 2);
  Nx = size(X, 1);
  Nm = size(M, 2);
  Nw = size(W, 1);
  Np = size(T,2) - 1;
  
  # Process cases with uncorrelated noise and a global transformation matrix 
  # apart as it allows to significantly fasten the code execution
  if(numel(size(W)) == 2 && numel(size(M)) == 2)
    chi2 = sumsq( W .* ( X - M * T * a ) );
    return;
  end
  
  # Compute the chi-square
  chi2 = zeros(1,N);
  for i=1:N
    Wi = gls_mean_getW(W,i);
    Mi = gls_mean_getM(M,i);
    chi2(i) = sumsq( Wi * ( X(:,i) - Mi * T * a(:,i) ) );
  end
end

################################################################################
# Demo                                                                         #
################################################################################

%!demo
%!  clear;
%! 
%!  # Problem size
%!   N = 64; # The number of observations
%!  Nx = 8;   # The number of variables in X, X has a size of ( Nx x N )
%!  Nw = 8;   # The number of variables in W, W's have sizes of ( Nw x Nx )
%!  Nm = 8;   # The number of variables in the mean, M has a size of ( Nx x Nm )
%!  Np = 0;   # The number of background components, P has a size of ( Nm x Np )
%!  Nn = 1024;   # The number of noise realization
%! 
%!  # Should we use correlated noise on the observations?
%!  correlated = true;
%! 
%!  # The fraction of observations to keep as a validation set
%!  fval = 0;
%! 
%!  # Should we use transformation matrices, M (see help of gls_mean) ?
%!  use_M = true;    # Set to true in order to use transformation matrices
%!  M_global = false; # Should all observations have the same M or should we use
%!                    # one M per observations
%! 
%!  # Minimal/Maximal SNR of each observation
%!  snrmin = 1;
%!  snrmax = 100;
%! 
%!  # Build the options to pass to gls_mean
%!  opts = gls_mean("options");
%!  opts.maxiter=1024;
%!  opts.chi2tol=1e-14;
%!  opts.n_no_improve=1;
%!  opts.rtol=1e-6;
%!  opts.rtolf=@(m,dm,rtol) max(abs(dm)./(abs(m)+abs(rtol)));
%!  opts.fbatch=1;
%!  opts.beta=0;
%!  opts.early_stop=0;
%!  opts.move2best_only=0;
%!  opts.verbose=0;
%!  opts.lambda=0;
%!  opts.lambda_cov=1e-14;
%!  opts.lsqrf = [];
%!  opts.seed = 1;
%! 
%!  # The random seed to use
%!  seed = 1;
%! 
%!  # Should we plot results?
%!  plotres = false;
%! 
%!  # Initialize the random number generators
%!  rand("seed", seed); randn("seed", seed);
%! 
%!  # Pick-up the background templates
%!  P = orth(vander([0:Nm-1]'/Nm, Np));
%! 
%!  # Pick-up the transformation matrices
%!  if(use_M)
%!    if(M_global)
%!      M = 2 * rand(Nx, Nm) - 1;
%!    else
%!      M = zeros(Nx,Nm,N);
%!      for i = 1:N
%!        M(:,:,i) = 2 * rand(Nx, Nm) - 1;
%!      end
%!    end
%!  else
%!    assert(Nx == Nm);
%!    M = eye(Nx,Nm);
%!  end
%! 
%!  # Pick-up the real mean value
%!  mreal = 2 * rand(Nm,1) - 1;
%!  if(~isempty(P))
%!    mreal = mreal - P * P' * mreal;
%!  end
%!  mreal = mreal / norm(mreal);
%! 
%!  # Pick-up real coefficients
%!  areal = 2 * rand(Np+1, N) - 1;
%! 
%!  # Build the real observations
%!  if( ~use_M || M_global )
%!    Xreal = M * [P mreal] * areal;
%!  else
%!    Xreal = zeros(Nx, N);
%!    for i=1:N
%!      Xreal(:,i) = M(:,:,i) * [P mreal] * areal(:,i);
%!    end
%!  end
%! 
%!  # Pick-up the SNR of each observations
%!  snr = ( snrmax - snrmin ) * rand(1,N) + snrmin;
%! 
%!  # Pick-up the weight matrices
%!  if(correlated)
%!    U = zeros(Nx,Nw,N);
%!    W = zeros(Nw,Nx,N);
%!    for i=1:N
%!      xstd = 0.001 + 0.999 * rand(Nw, 1);
%!      xstd = ( xstd / norm(xstd) ) * ( norm(Xreal(:,i)) / snr(i) );
%!      U(:,:,i) = orth(2 * rand(Nx, Nw) - 1) * diag(xstd);
%!      W(:,:,i) = pinv(U(:,:,i));
%!    end
%!  else
%!    assert(Nx == Nw);
%!    Xstd = 0.001 + 0.999 * rand(Nx,N);
%!    Xstd = ( Xstd ./ norm(Xstd, "cols") ) .* ( norm(Xreal,"cols") ./ snr );
%!    W = 1 ./ Xstd;
%!  end
%!  
%!  # Get the validation observations
%!  ival = randperm(N, floor(fval * N));
%! 
%!  # Perform nnoise noise realization
%!  if(Nn > 0)
%!    m = zeros(Nm, Nn);
%!    chi2 = zeros(Nn,1);
%!    C = zeros(Nm, Nm, Nn);
%!    iter = zeros(Nn,1);
%!    for inoise=1:Nn
%!      if(correlated)
%!        for i=1:N
%!          X(:,i) = Xreal(:,i) + U(:,:,i) * randn(Nw, 1);
%!        end
%!      else
%!        X = Xreal + Xstd .* randn(Nx, N);
%!      end
%!      [mi, chi2i, Ci, iteri] = gls_mean(X, W, M, P, [], ival, opts);
%!      mi_scale = sign(mreal' * mi) * norm(mi);
%!      m(:,inoise) = mi / mi_scale;
%!      iter(inoise) = iteri;
%!      C(:,:,inoise) = Ci / mi_scale^2;
%!      chi2(inoise) = chi2i;
%!      printf("\rComputing %d noise realization: %d done", Nn, inoise);
%!    end
%!    printf("\n");
%! 
%!    # Get the observed and median predicted covariance matrices
%!    Cobs = cov(m');
%!    Cpred = mean(C,3);
%! 
%!    # Get the observed and predicted uncertainties
%!    merrobs = sqrt(diag(Cobs));
%!    merrpred = sqrt(diag(Cpred));
%! 
%!    # Get the observed and theoretical correlation matrices
%!    Corrobs =  diag(1./merrobs) *  Cobs * diag(1./merrobs);
%!    Corrpred = diag(1./merrpred) * Cpred * diag(1./merrpred);
%! 
%!    # Print results
%!    dm = abs( ( m - mreal ) ./ mreal );
%!    dmerr = abs( ( merrobs - merrpred ) ./ merrobs );
%!    dC = abs( Corrobs - Corrpred );
%!    stat_str = @(x) sprintf("mean=%g, median=%g, std=%g, min=%g, max=%g", mean(x), median(x), std(x), min(x), max(x));
%!    printf("Absolute relative error in m: %s\n", stat_str(dm(:)));
%!    printf("Absolute relative error in merr: %s\n", stat_str(dmerr(:)));
%!    printf("Absolute error in the correlation coefficients: %s\n", stat_str(dC(:)));
%!    printf("Reduced chi-square: %s\n", stat_str(chi2));
%!    printf("Number of iterations: %s\n", stat_str(iter));
%!  end
%! 
%!  # Plot results if needed
%!  if(plotres)
%!    # Reset verbosity
%!    opts.verbose=true;
%! 
%!    # Perform a last fit for plotting
%!    if(correlated)
%!      for i=1:N
%!        X(:,i) = Xreal(:,i) + U(:,:,i) * randn(Nw, 1);
%!      end
%!    else
%!      X = Xreal + Xstd .* randn(Nx, N);
%!    end
%!    [m, ~, C] = gls_mean(X, W, M, P, [], ival, opts);
%!    m_scale = sign(mreal' * m) / norm(m);
%!    m = m * m_scale;
%!    C = C * m_scale^2;
%!    merr = sqrt(diag(C));
%! 
%!    # Plot the mean vector
%!    printf("Plotting the mean observations... Press 'q' to quit...\n");
%!    plot(mreal,"k-;Real;",m,"r-;Found;", m-merr, "r:;Uncertainties;", m+merr, "r:");
%!    title("Mean");
%!    k=kbhit();
%!    if(k == 'q' || k =='Q')
%!      return;
%!    end
%! 
%!    # Plot observations
%!    printf("Plotting observations... Press 'q' to quit...\n");
%!    for i=1:N
%!      # Get Wi, Mi
%!      if(correlated)
%!        Wi = W(:,:,i);
%!      else
%!        Wi = diag(W(:,i));
%!      end
%!      if(~use_M || M_global)
%!        Mi = M;
%!      else
%!        Mi = M(:,:,i);
%!      end
%! 
%!      # Compute the final solution (because we normalized m)
%!      a = ( Wi * Mi * [P m] ) \ ( Wi * X(:,i) );
%! 
%!      # Compute the reduced chi-square associated with this observation
%!      chi2i = sumsq( Wi * ( X(:,i) - Mi * [ P m ] * a ) ) / ( rank(Wi) - (Np + 1) );
%!      if(Np > 0)
%!        plot(Xreal(:,i), "k-;Real signal;", ...
%!             X(:,i), "kd;Observation;", ...
%!             Mi * P * a(1:Np), "b-;Found background;",  ...
%!             Mi * P * areal(1:Np,i), "b:;Real background;",  ...
%!             Mi * [ P m ] * a, "r-;Found signal;");
%!      else
%!        plot(Xreal(:,i), "k-;Real signal;", ...
%!             X(:,i), "kd;Observation;",
%!             Mi * m * a, "r-;Found signal;");
%!      end
%!      title(sprintf("Observation %d (chi2r: %f, snr: %f)",i,chi2i,snr(i)));
%!      k = kbhit();
%!      if(k == 'q' || k =='Q')
%!        return;
%!      end
%!    end
%!  end
