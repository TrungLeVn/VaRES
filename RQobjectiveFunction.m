function output = RQobjectiveFunction(BETA, OUT, MODEL, y, THETA, empiricalQuantile)                                                                                                       
% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
% *                                                                                                                                      *
% * By SIMONE MANGANELLI, European Central Bank, Frankfurt am Ma.                                                                      *
% * Created on 15 September, 2000. Last modified 20 February 2002.                                                                       *
% *                                                                                                                                      *
% ****************************************************************************************************************************************
% 
% 
% RQobjectiveFunction computes the VaR and the RQ criterion for the vector of parameters BETA, for MODEL i (1:SAV, 2:AS, 3:GARCH, 4:ADAPTIVE) 
% given the number of observations T, the time series y and the confidence level THETA.
%
% If OUT=1, the output is the regression quantile objective function.
% If OUT=2, the output is [VaR, Hit].
%
%**********************************************
% Compute the VaR
%% Initial Conditions
T = length(y);
VaR = zeros(T,1);
Hit = VaR;

VaR(1) = empiricalQuantile;
Hit(1) = THETA - (y(1) < VaR(1));
%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1

   VaR = SAVloop(THETA, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   %VaR = SAVloop(BETA,y,VaR(1));
   Hit = THETA - (y < VaR);

%
%********************************************************************************************
% Model 2: Asymmetric Slope.
%
else
    
   VaR = ASloop(THETA, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   Hit = THETA - (y < VaR);
end

RQ  = Hit'*(y - VaR);

if RQ == Inf || (RQ ~= RQ) || ~isreal(RQ)
   RQ = 1e+100;
end
%
%**********************************************
% Select the output of the program.
if OUT == 1
    output = RQ;
elseif OUT ==2
    output = [VaR, Hit];
else
    error('Wrong output selected. Choose OUT = 1 for RQ, or OUT = 2 for [VaR, Hit].')
end
end
