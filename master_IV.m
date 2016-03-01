global  n y x z m

Rep = 500; % # of Monte Carlo replication

n = 120; % n takes from [120, 240]
m = 80; % m takes from [80, 160]

seed = 301;
useful = 4; % the number of useful (non-zero) x's
rho = 0.6; % the endogeneity
beta0 = [1;1]; % column vector. Good.


beta_lasso_2sls = zeros(2, Rep);
beta_lasso_fuller = zeros(2, Rep);
%beta_hat_RJIVE = zeros(2, Rep);

tic
for r = 1:Rep
    
    seed_v = seed + r;
    rng(seed_v);
    
    [y, x, z] = dgpLinearIV(beta0, rho, useful); % generate the data
    
    % with the generated data
    % use the post-lasso and Hansen Kozbur's RJIVE to estimate the
    % parameter
	[beta_lasso_2sls(:, r),~,beta_lasso_fuller(:,r),~,~] = post_lasso(y, x, z);
    
    %beta_hat_RJIVE(:, r) = RJIVE(y, x, z);

end
toc