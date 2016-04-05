function [ beta ] = Rliml( y, x, Z )

global m n
[~,p] = size(x);
alpha_grid = (0.01:0.01:0.05)'; 
T = length(alpha_grid);
K_n = Z' * Z / n;
[ve, va] = eig(K_n);
va = sum(va);
va = va(m:-1:1);
ve(:,1:m) = ve(:,m:-1:1);
va2 = va.^2;
psi = zeros(n,m);
for k = 1:m
    psi(:,k) = Z * ve(:,k) * sqrt( va(k) )^(-1);
end
tra = zeros(T,2);
stl = zeros(T,1);
COV = zeros(p,p,T);
options = optimoptions('fmincon','Algorithm','interior-point','MaxIter',5000,'MaxFunEvals',5000);

%LIML%
P = Z * inv(Z'*Z) * Z';
b_2sls = (x' * P * x) \ (x' * P * y);
f = @(y,x,P,d) ( (y - x*d)' * P * (y - x*d) )/( (y - x*d)' * (y - x*d) );

[~, nu] = fmincon(@(d)f(y,x,P,d), b_2sls,[], [], [], [], -4*ones(2,1), 6*ones(2,1),[], options);
b_liml = (x' * ( P - nu*eye(n) ) * x) \ (x' * ( P - nu*eye(n) ) * y) ;

%Tikhonov
ind = 1;
for t = 1:T
    alpha = alpha_grid(t);
    vv = (va2 + alpha).^(-1);
    vc = repmat( va2 .* vv, n, 1 );
    Pa = ( psi .* vc ) * psi' / n;
    u=(eye(n)-Pa) * x;
    tra(t,:) = [ sum(diag(Pa)), sum( diag( Pa*Pa ) ) ];
    %The case in Carrasco and Tchuente(2015) is a scalar case, the variance
    %is easy to compare. Now we consider two endogenous variables, the
    %covariance is a 2 by 2 matrix. We compare two matrices A and B by
    %determine whether A - B is a positive definite matrix.
    COV(:,:,t) = (u' * u / n) / (1 - sum( diag(Pa) ) / n)^2;
    if t == 1
        cvt_min = COV(:,:,t);
    else
        [~,pvedef] = chol( cvt_min - COV(:,:,t) );
        if pvedef == 0
            cvt_min = COV(:,:,t);
            ind = t;
        end
    end
end

alpha = alpha_grid(ind);
vd = (va2 + alpha).^(-1);
vc1 = repmat( va2 .* vd, n, 1);
Pa1 = (psi .* vc1) * psi' / n;
u = (eye(n) - Pa1) * x;
sigu2 = u' * u / n;
dh = inv(x' * Pa1 * x) * (x' * Pa1 * y);
sigeps2 = (y - x * dh)' * (y - x * dh) / n;
sigueps = u' * (y - x * dh) / n;
v = u - (y - x * dh) * (sigueps / sigeps2)';
sigv2 = v' * v / n;

for t = 1:T;
    STL = sigeps2 * (sigv2 * tra(t,2) / n + COV(:,:,t)- sigu2 * tra(t,2) / n);
    stl(t) = max( eig(STL) );
end;

[~, ind] = min(stl);
alpha_opt_l = alpha_grid(ind);

vol=(va2 + alpha_opt_l).^(-1);
vc2l = repmat( va2 .* vol, n, 1);
Paol = (psi .* vc2l) * psi' / n;    %Kn_alpha inverse
[~, mut] = fmincon(@(d)f(y,x,Paol,d), b_liml,[], [], [], [], -4*ones(2,1), 6*ones(2,1),[], options);
beta_Rliml_T =  (x' * (Paol - mut*eye(n) ) * x) \ (x'* (Paol - mut*eye(n) ) * y);
beta = beta_Rliml_T(1);
end