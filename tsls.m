function [b VC1 VC2] = tsls(y,d,x,z,VG)
% [b VC1 VC2] = tsls(y,d,x,z,VG) computes 2sls estimate of coefficients b and
% variance covariance matrix VC assuming homoskedasticity for outcome 
% variable y where d are endogenous variables, in structural equation,
% x are exogensous variables in structural equation and z are 
% instruments.  x should include the constant term.
% 
% If the optional argument VG is supplied, the variance covariance matrix
% assuming that sqrt(n) times the unidentified regression coefficient on z 
% is a normal random variable with mean 0 and variance VG is computed as VC2.


n = size(y,1);
k = size(x,2) + size(d,2);

X = [d x];
Z = [z x];

Mxz = X'*Z;
Mzz = inv(Z'*Z);
M   = inv(Mxz*Mzz*Mxz'); %#ok<*MINV>

b = M*Mxz*Mzz*(Z'*y);
e = y - X*b;

VC1 = (e'*e/(n - k))*M;

if nargin > 4,
    VC2 = VC1 + M*Mxz*VG*Mxz'*M;
end
