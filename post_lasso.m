function [beta_2sls, var_2sls] = post_lasso(y,x,Z)

%This function implement the post-lasso estimation procedure proposed by
%BCCH (2012).

global n m;

lambda0C = 2.2*sqrt(2*log(2*m*log(n)/.1)); %tuning parameter

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
%take union to find the index of all selected instruments and save into
%ind0
ind0 = logical( sum(Ind,2) );
Z1 = Z(:,logical(ind0));

%Run 2-stage-least-square
if isempty(Z1),
    beta_2sls = NaN;
    var_2sls = NaN;
else
    [btemp1,VCtemp1] = tsls(y,x,[],Z1);
    beta_2sls = btemp1;
    var_2sls = VCtemp1;
end

end