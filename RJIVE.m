function [ beta_rjive ] = RJIVE( y, x, Z )

%This function implements the estimation procedure proposed by Hansen and
%Kozbur(2014), the regularized JIVE estimator.
global n m;
dim_x = size(x,2);

C = std( (eye(n) - x * inv(x' * x) * x') * y ); %compute the const C

Lambda_square = C^2 * m * eye(m);
%based on the closed form of the estimator: beta_hat = ( F1 )^(-1) * F2
F1 = zeros(dim_x, dim_x);
F2 = zeros(dim_x, 1);

for i = 1:n
    Z_i = Z(i,:)';
    x_i = x(i,:)';
    y_i = y(i);
    PI_i = (Z'*Z - Z_i*Z_i' + Lambda_square) \ (Z'* x - Z_i*x_i');
    F1 = F1 + PI_i' * Z_i * x_i';
    F2 = F2 + PI_i' * Z_i * y_i;
end

beta_hat = F1 \ F2;
beta_rjive = beta_hat(1);

end

