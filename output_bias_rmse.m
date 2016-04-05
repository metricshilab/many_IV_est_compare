function [ bias, RMSE, num_outlier ] = output_bias_rmse( beta, beta0)

%This function is designed to calculate the bias and RMSE for the
%estimation result. Since there is some chances for the dgp to generate some
%troublesome data, the estimation result may contain NaN,Inf or some outliers
%far from the true value we deal with the toublesome cases in this function.

global Rep

outlier_std = 15; %tuning parameter: the standard to determine whether the result is an outlier 

if sum( isnan(beta) ) || sum( isinf(beta) ) || sum( (abs( beta - beta0 ) > outlier_std) )
    trouble_ind = logical( isnan(beta) + isinf(beta)  + (abs(beta - beta0) > outlier_std) );
    Ind = logical( ones(Rep,1) - trouble_ind );%index of troublesome results
    bias = mean( beta(Ind) ) - beta0;
    RMSE = sqrt( mean( (beta(Ind) - beta0) .^2 ) );
    num_outlier = sum( trouble_ind );
else
    bias = mean( beta ) - beta0;
    RMSE = sqrt( mean( (beta - beta0) .^ 2) );
    num_outlier = 0;
end


end

