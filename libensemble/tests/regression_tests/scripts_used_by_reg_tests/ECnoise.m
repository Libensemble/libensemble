function [fnoise,level,inform] = ECnoise(nf,fval)
%
%  Determines the noise of a function from the function values
%
%     [fnoise,level,inform] = ECnoise(nf,fval)
%
%  The user must provide the function value at nf equally-spaced points.
%  For example, if nf = 7, the user could provide
%
%     f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)
%
%  in the array fval. Although nf >= 4 is allowed, the use of at least
%  nf = 7 function evaluations is recommended.
%
%  Noise will not be detected by this code if the function values differ
%  in the first digit.
%
%  If noise is not detected, the user should increase or decrease the
%  spacing h according to the output value of inform.  In most cases,
%  the subroutine detects noise with the initial value of h.
%
%  On exit:
%    fnoise is set to an estimate of the function noise;
%       fnoise is set to zero if noise is not detected.
%
%    level is set to estimates for the noise. The k-th entry is an
%      estimate from the k-th difference.
%
%    inform is set as follows:
%      inform = 1  Noise has been detected.
%      inform = 2  Noise has not been detected; h is too small.
%                  Try 100*h for the next value of h.
%      inform = 3  Noise has not been detected; h is too large.
%                  Try h/100 for the next value of h.
%
%     Argonne National Laboratory
%     Jorge More' and Stefan Wild. November 2009.
disp(nf)
disp(fval)

level = zeros(nf-1,1);
dsgn  = zeros(nf-1,1);
fnoise = 0.0;
gamma = 1.0; % = gamma(0)

% Compute the range of function values.
fmin = min(fval);    fmax = max(fval);
if (fmax-fmin)/max(abs(fmax),abs(fmin))>.1;
    inform = 3; return;
end

% Construct the difference table.
for j = 1:nf-1
    for i = 1:nf-j
        fval(i) = fval(i+1) - fval(i);
    end

    % h is too small only when half the function values are equal.
    if (j==1 && sum(fval(1:nf-1)==0)>=nf/2)
        inform = 2; return;
    end

    gamma = 0.5*(j/(2*j-1))*gamma;

    % Compute the estimates for the noise level.
    level(j) = sqrt(gamma*mean(fval(1:nf-j).^2));

    % Determine differences in sign.
    emin = min(fval(1:nf-j));
    emax = max(fval(1:nf-j));
    if (emin*emax < 0.0); dsgn(j) = 1; end
end

% Determine the noise level.
for k = 1:nf-3
    emin = min(level(k:k+2));
    emax = max(level(k:k+2));
    if (emax<=4*emin && dsgn(k))
        fnoise = level(k);
        inform = 1;
        return;
    end
end

% If noise not detected then h is too large.
inform = 3;

return;
