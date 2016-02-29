function [b VC VCb VCc] = fuller(y,d,x,z,c)
% [b VC VCb VCc] = fuller(y,d,x,z,c) computes FULLER estimate of coefficients b and
% variance covariance matrix using usual asymptotic approximation (VC) and
% Bekker asymptotic approximation (VCb) assuming homoskedasticity for outcome 
% variable y where where d are endogenous variables, in structural equation,
% x are exogensous variables in structural equation and z are 
% instruments.  x should include the constant term.  c is a user specified
% parameter (c = 1 higher order unbiased; c >= 4 higher order admissible
% under quadratic loss)
% 


n = size(y,1);
k = size(x,2) + size(d,2);

Mxinv = inv(x'*x);
if ~isempty(x),
    My    = y - x*Mxinv*(x'*y); %#ok<*MINV>
    Md    = d - x*Mxinv*(x'*d);
    Mz    = inv(z'*z - (z'*x)*Mxinv*(x'*z));
else
    My = y;
    Md = d;
    Mz = inv(z'*z);
end
alpha = min(eig(full(inv([My Md]'*[My Md])*(([My Md]'*z)*...
          Mz*(z'*[My Md])))));
alpha1 = (alpha - (1 - alpha)*c/(n - k - size(z,2)))/(1 - (1 - alpha)*c/(n - k - size(z,2)));      

X     = [d x];
Z     = [z x];
Mxy   = X'*y;
Mzy   = Z'*y;
Mzx   = Z'*X;
Mxx   = X'*X;
Mzz   = inv(Z'*Z);
Mxzzx = Mzx'*Mzz*Mzx;

H     = Mxzzx - alpha1*Mxx;
b     = inv(H)*(Mzx'*Mzz*Mzy - alpha1*Mxy);
e     = y - X*b;

J     = Mxzzx - alpha1*(X'*e)*(e'*X)/(e'*e);
S     = (1 - alpha1)*J - alpha1*H;

VC    = (e'*e/(n - k))*inv(H);
VCb   = (e'*e/(n - k))*inv(H)*S*inv(H);

s2 = (e'*e)/(n-k);
atilde = (e'*Z)*Mzz*(Z'*e)/(e'*e);
Ups = Z*Mzz*(Z'*X);
Xhat = X - e*(e'*X)/(e'*e);
Vhat = Xhat - Z*Mzz*(Z'*Xhat);
tau = k/n;
kappa = 0;
A1 = 0;
A2 = 0;
B1 = 0;
for ii = 1:n
    pii = Z(ii,:)*Mzz*Z(ii,:)';
    kappa = kappa + pii;
    A1 = A1+(pii-tau)*Ups(ii,:);
    A2 = A2+(e(ii,1)^2)*Vhat(ii,:)/n;
    B1 = B1+((e(ii,1)^2-s2)*(Vhat(ii,:)'*Vhat(ii,:)));
end
kappa = kappa/k;
B = k*(kappa-tau)*B1/(n*(1-2*tau+kappa*tau));
A = A1'*A2;
SB = s2*(((1-atilde)^2)*((Xhat'*Z)*Mzz*(Z'*Xhat))...
    +(atilde^2)*((Xhat'*Xhat)-(Xhat'*Z)*Mzz*(Z'*Xhat)));
VCc = inv(Mxzzx - atilde*Mxx)*(SB+A+A'+B)*inv(Mxzzx - atilde*Mxx);



