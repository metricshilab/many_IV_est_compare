function [beta_2sls, var_2sls, beta_fuller, var_fuller,FS] = post_lasso(y,x,Z)

global n m;

lambda0C = 2.2*sqrt(2*log(2*m*log(n)/.1));
dim_x = size(x,2);
%Save the index of optimal instruments for both x1 and x2
Ind = zeros(m,dim_x);

%For both x1, x2, select optimal instruments by lasso estimator
for i = 1:dim_x
    
    e0 = x(:,i);
    Ups0 = sqrt((e0.^2)'*(Z.^2))';
    coefTemp = LassoShooting2(Z, x(:,i), lambda0C*Ups0, 'verbose', 0);
    ind0 = (abs(coefTemp) > 0);
    Z0 = Z(:,ind0);

    for mm = 1:15
        e1 = x(:,i)- Z0 * (Z0 \ x(:,i));
        Ups1 = sqrt((e1.^2)'*(Z.^2))';
        coefTemp = LassoShooting2(Z,x(:,i),lambda0C*Ups1,'verbose',0);
        ind1 = (abs(coefTemp) > 0);
        Z0 = Z(:,ind1);
    end
    
    Ind(:,i) = ind1;
    
end

ind0 = zeros(m,1); %save the index of all the instruments that are selected

%take union
for j = 1:dim_x
    ind0 = (ind0 + Ind(:,j) ) - (ind0.*Ind(:,j));
end

Z1 = Z(:,logical(ind0));

if isempty(Z1),
    beta_2sls = NaN;
    var_2sls = NaN;
    beta_fuller = NaN;
    var_fuller= NaN;
    FS = 0;
else
    bfs = Z1\x;
    efs = x - Z1*bfs;
    Vfs = ((Z1'*Z1)\((Z1.*((efs.^2)*ones(dim_x,sum(ind0))))'*Z1))/(Z1'*Z1);
    FS = bfs'*(Vfs\bfs)/sum(ind0);
    [btemp1,VCtemp1] = tsls(y,x,[],Z1);
    [btemp2,~,~,VCtemp2] = fuller(y,x,[],Z1,1);
    beta_2sls = btemp1;
    var_2sls = VCtemp1;
    beta_fuller = btemp2;
    var_fuller = VCtemp2;
end

end