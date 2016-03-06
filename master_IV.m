global  n y x z m

Rep = 500; % # of Monte Carlo replication
n_choice = [120; 240];
m_choice = [80; 160];
p = size(n_choice,1);
q = size(m_choice,1);

seed = 301;
useful = 4; % the number of useful (non-zero) x's
rho = 0.6; % the endogeneity
beta0 = [1;1]; % column vector. Good.

%set up variables to store the estimation result
beta_lasso_2sls = zeros(Rep,p,q);
beta_RJIVE = zeros(Rep,p,q);
bias_lasso_2sls = zeros(p,q);
bias_RJIVE = zeros(p,q);
RMSE_lasso_2sls = zeros(p,q); 
RMSE_RJIVE = zeros(p,q);


for i = 1:p
    n = n_choice(i);
    for j = 1:q
        m = m_choice(j);
        tic
        for r = 1:Rep
            %Fix the seed for each iteration
            disp(r);
            seed_v = seed + r;
            rng(seed_v);
            %Generating data
            [y, x, z] = dgpLinearIV(beta0, rho, useful); % generate the data

            % with the generated data
            % use the post-lasso and Hansen Kozbur's RJIVE to estimate the
            % parameter
            [b_lasso_temp,var_lasso_temp] = post_lasso(y, x, z);
            beta_lasso_2sls(r,i,j) = b_lasso_temp(1); 
            beta_RJIVE(r,i,j) = RJIVE(y, x, z);
        end
        toc
        %Handle troublesome cases, i.e. the estimation result is inf or
        %nan, and compute the bias and RMSE for each combination of n, m
        
        %For post_lasso
        if sum( isnan(beta_lasso_2sls(:,i,j) ) ) || sum( isinf(beta_lasso_2sls(:,i,j) ) )
            beta_temp = beta_lasso_2sls(:,i,j);
            trouble_ind = logical( isnan(beta_temp) + isinf(beta_temp) );
            Ind = logical( ones(Rep,1) - trouble_ind );
            sum( beta_temp(Ind) )
            bias_lasso_2sls(i,j) = mean( beta_temp(Ind) ) - beta0(1);
            RMSE_lasso_2sls(i,j) = sqrt( mean( (beta_temp(Ind) - 1) .^2 ) );
        else
            bias_lasso_2sls(i,j) = mean( beta_lasso_2sls(:,i,j) ) - beta0(1);
            RMSE_lasso_2sls(i,j) = sqrt( mean( (beta_lasso_2sls(:,i,j) - 1) .^ 2) );
        end
        %For RJIVE
        if sum( isnan(beta_RJIVE(:,i,j) ) ) || sum( isinf(beta_RJIVE(:,i,j) ) )
            beta_temp = beta_RJIVE(:,i,j);
            trouble_ind = logical( isnan(beta_temp) + isinf(beta_temp) );
            Ind = logical( ones(Rep,1) - trouble_ind );
            sum( beta_temp(Ind) )
            bias_RJIVE(i,j) = mean( beta_temp(Ind) ) - beta0(1);
            RMSE_RJIVE(i,j) = sqrt( mean( (beta_temp(Ind) - 1) .^2 ) );
        else
            bias_RJIVE(i,j) = mean( beta_RJIVE(:,i,j) ) - beta0(1);
            RMSE_RJIVE(i,j) = sqrt( mean( (beta_RJIVE(:,i,j) - 1) .^ 2) );
        end
    end
end