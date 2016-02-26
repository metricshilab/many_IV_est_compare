global  n y x z m

Rep = 5; % # of Monte Carlo replication

n = 100;
m = 80;

seed = 301;
useful = 4; % the number of useful (non-zero) x's
rho = 0.6; % the endogeneity
beta0 = [1;1]; % column vector. Good.


tic
r = 1;

while r <= Rep
    seed_v = seed + r;
    rng(seed_v);
    
    [y, x, z] = dgpLinearIV(beta0, rho, useful); % generate the data
    
    % with the generated data
    % use the post-lasso and Hansen Kozbur's RJIVE to estimate the
    % parameter
    
    
    r = r + 1;
    toc
end



