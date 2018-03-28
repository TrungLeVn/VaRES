function [BetaHat, VaR, output] = CAViaR_X(y,varargin)

% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
% *                                                                                                                                      *
% * Orignal by SIMONE MANGANELLI, European Central Bank, Frankfurt am Main.                                                                      *
% * Modify by Trung Le by using the new bandwidth from the VAR for VaR paper
% * Function now can be used for any choice of quantiles and regressors
% * other than the absolute return
% *                                                                                                                                      *
% ****************************************************************************************************************************************
quantileDefault = 0.05;
periodDefault = 1;
nlagDefault = 0;
callerName = 'CAViaR_X';
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'scalar','>',0,'<',1},callerName));
addParameter(parseObj,'Model',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'NumLags',nlagDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer'},callerName));
addParameter(parseObj,'Ovlap',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'X',[],@(x)validateattributes(x,{'numeric'},{'2d'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'xDates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'GetSe',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Display',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Options',[],@(x)validateattributes(x,{},{},callerName));
addParameter(parseObj,'Params',[],@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'DoParallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Cores',4,@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'Constrained',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'DoFull',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'startPars',[]);

parse(parseObj,varargin{:});
theta = parseObj.Results.Quantile;
model = parseObj.Results.Model;
period = parseObj.Results.Period;
nlag = parseObj.Results.NumLags;
ovlap = parseObj.Results.Ovlap;
Regressor = parseObj.Results.X;
yDates = parseObj.Results.Dates;
xDates = parseObj.Results.xDates;
options = parseObj.Results.Options;
getse = parseObj.Results.GetSe;
display = parseObj.Results.Display;
BetaHat = parseObj.Results.Params; 
doparallel = parseObj.Results.DoParallel;
cores = parseObj.Results.Cores;
constrained = parseObj.Results.Constrained;
dofull = parseObj.Results.DoFull;
startPars = parseObj.Results.startPars;

if ~ismember(model,[1,2])
    disp(' ')
    disp('*******************************************************************')
    disp('ERROR! You need to select one of the following models:')
    disp('Model=1: Symmetric Absolute Value')
    disp('Model=2: Asymmetric Slope')
end

%%
% Organize data according to the return horizon chosen by the period
% argument

% Replace missing value by the mean of the in-sample estimation
y = y(:);
y(isnan(y)) = nanmean(y);
nobs = length(y);


% Load dates
if isempty(yDates)
    yDates = 1:nobs;
else
    if iscell(yDates)
    yDates = datenum(yDates);    
    end
    if numel(yDates) ~= nobs
    error('Length of Dates must equal the number of observations.')
    end
end

if isempty(xDates)
    xDates = yDates;
end
% Load the conditioning variable (predictor)
if isempty(Regressor)
   error('Regressor must be provided when using CAViaR-X.')
end

% Mix the data to fit the return horizon and nlag similar to the MIDAS model
if ~dofull % If we need to rescale the series to appropriate horizons
MixedData = MixedFreqQuantile(y,yDates,y,yDates,nlag,period,ovlap);
y = MixedData.EstY;
yDates = MixedData.EstYdate;
nobs = length(y);
MixedVar = MixedFreqQuantile(Regressor,xDates,Regressor,xDates,nlag,period,ovlap);
Regressor = sqrt(MixedVar.EstY);
end

% Start parallel if allowed and not having working parallel
if doparallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local',cores);
    end
end

rng('default')
rng(2706); % To ensure reproducity of estimations

%%
% Compute the empirical THETA-quantile for y (the in-sample vector of observations)
   if period == 1
       InitialEmp = 300;
   elseif period == 5
       InitialEmp = 100;
   else
       InitialEmp = 50;
   end
   % NOTE: It seems that the estimation of CAViaR model with symmetric
   % structure sensitive to the choice of how many initial observation
   % (Multi-period horizons) are chosen to compute the initial empirical
   % VaR
   ysort          = sortrows(y(1:InitialEmp), 1); 
   empiricalQuantile = ysort(round(InitialEmp*theta));

%%
% if the BetaHat is provided, just compute the VaR and Hit percentage value
if ~isempty(BetaHat)
    display = false;
     % Compute VaR and Hit for the estimated parameters of RQ.
    VaRHit  = RQobjectiveFunction(BetaHat', 2, model, y,Regressor, theta, empiricalQuantile);
    VaR = VaRHit(:,1);
    Hit = VaRHit(:,2);
    
    % Compute the percentage of hits in sample and out-of-sample.
    HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
    if nargout > 2
    output.beta = BetaHat;
    output.VaR = VaR;
    output.Hit = Hit;
    output.Dates = yDates;
    output.HitPercentage = HitPercentage;
    output.y = y;
    end
else
%%
% *****************************************************************************************
% Set parameters for optimisation.
% *****************************************************************************************
REP			  = 10;                % Number of times the optimization algorithm is repeated.
if isempty(startPars)
fprintf('Finding the initial Betas... \n');
if (model == 1)
    nInitialVectors = [10000, 4]; % Number of initial vector fed in the uniform random number generator SAV and GARCH models.
    nInitialCond = 15;             % Select the number of initial conditions for the optimisation.
    
else
    nInitialVectors = [100000, 5]; % Number of initial vector fed in the uniform random number generator for AS model.
    nInitialCond = 15;            % Select the number of initial conditions for the optimisation.
end
end
MaxFunEvals = 1000; % Parameters for the optimisation algorithm. Increase them in case the algorithm does not converge.
MaxIter     = 1000;
if isempty(options)
options = optimset('LargeScale', 'off', 'HessUpdate','dfp', 'MaxFunEvals', MaxFunEvals, ...
                    'display', 'off', 'MaxIter', MaxIter, 'TolFun', 1e-10, 'TolX', 1e-7);
end

% Some meaningful constrains for the paramters space - To be used in
% fminsearchbnd
   tol = 1e-7;
   if model == 1 
      lb = [-Inf;0+tol;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf];
   else
      lb = [-Inf;0+tol;-Inf;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf;Inf];
   end
     
optionsUnc = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton',...
    'MaxFunEvals',MaxFunEvals,'MaxIter',MaxIter);

optionCon = optimoptions('fmincon', 'MaxFunEvals', MaxFunEvals, ...
                    'display', 'off', 'MaxIter', MaxIter);
warning off


%**************************** Start the optimization ******************************************
    if isempty(startPars)
    initialTargetVectors = unifrnd(0, 1, nInitialVectors);
    
    RQfval = zeros(nInitialVectors(1), 1);
    if doparallel
        parfor i = 1:nInitialVectors(1)
        RQfval(i) = RQobjectiveFunction(initialTargetVectors(i,:), 1, model,y, Regressor, theta, empiricalQuantile);
        end
    else
        for i = 1:nInitialVectors(1)
        RQfval(i) = RQobjectiveFunction(initialTargetVectors(i,:), 1, model,y, Regressor, theta, empiricalQuantile);
        end
    end
    Results          = [RQfval, initialTargetVectors];
    SortedResults    = sortrows(Results,1);
    
    if (model == 1)
        BestInitialCond  = SortedResults(1:nInitialCond,2:5);
    else
        BestInitialCond  = SortedResults(1:nInitialCond,2:6);
    end
    else
        BestInitialCond = startPars';
    end
    fprintf('Optimizing parameters.... \n');
    if doparallel
    parfor i = 1:size(BestInitialCond,1)
        [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),BestInitialCond(i,:),options);
        for it = 1:REP
            try
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminunc(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),optionsUnc);    
            catch
            warning('fminunc does work. Move on to the fminsearch.');
            end
            % It seems to get better results with the fminsearch since the
            % ObjFunction is very nonlinear.
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),options);            
            if constrained
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearchbnd(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),lb,ub,options);      
              %if sum(Beta(i,:)' > lb) + sum(Beta(i,:)' < ub) ~= (2*length(Beta(i,:))) % If fminsearch violate the bounds, redo the optimization with fmincon
              %     [Beta(i,:), fval(i,1), exitflag(i,1)] = fmincon(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),[],[],[],[],lb,ub,[],optionCon);
              %end
            end
            if exitflag(i,1) == 1
                break
            end
        end          
    end
    else
        for i = 1:size(BestInitialCond,1)
        [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),BestInitialCond(i,:),options);
        for it = 1:REP
            try
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminunc(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),optionsUnc);    
            catch
            warning('fminunc does work. Move on to the fminsearch.');
            end
            % It seems to get better results with the fminsearch since the
            % ObjFunction is very nonlinear.
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),options);            
            if constrained
              %[Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearchbnd(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),lb,ub,options);      
              if sum(Beta(i,:)' > lb) + sum(Beta(i,:)' < ub) ~= (2*length(Beta(i,:))) % If fminsearch violate the bounds, redo the optimization with fmincon
                   [Beta(i,:), fval(i,1), exitflag(i,1)] = fmincon(@(BETA) RQobjectiveFunction(BETA,1, model, y, Regressor, theta, empiricalQuantile),Beta(i,:),[],[],[],[],lb,ub,[],optionCon);
              end
            end
            if exitflag(i,1) == 1
                break
            end
        end          
        end    
    end
    SortedFval  = sortrows([fval, Beta, exitflag, BestInitialCond], 1);
     if (model == 1)
        BestFval         = SortedFval(1, 1);
        BetaHat   = SortedFval(1, 2:5)';
        ExitFlag         = SortedFval(1, 6);
        InitialCond = SortedFval(1, 7:9)';
    else
        BestFval         = SortedFval(1, 1);
        BetaHat    = SortedFval(1, 2:6)';
        ExitFlag         = SortedFval(1, 7);
        InitialCond = SortedFval(1, 8:11)';     
     end
 %**************************** End of Optimization Routine ******************************************
%%

%************************** Compute variables that enter output *****************************

    % Compute VaR and Hit for the estimated parameters of RQ.
    VaRHit  = RQobjectiveFunction(BetaHat', 2, model, y,Regressor, theta, empiricalQuantile);
    VaR = VaRHit(:,1);
    Hit = VaRHit(:,2);
    nobs = length(VaR);
    % Compute the percentage of hits in sample and out-of-sample.
    HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
    if getse
    % Compute the variance-covariance matrix of the estimated parameters.
    [varCov] = VarianceCovariance(BetaHat', model, nobs, y, Regressor, theta, VaR);
    standardErrors = sqrt(diag(varCov));
    coeffPvalue    = normcdf(-abs(BetaHat ./ standardErrors));
    else
    standardErrors = nan(length(BetaHat),1);
    coeffPvalue = nan(length(BetaHat),1);
    end
    RQ       = BestFval;
    EXITFLAG = ExitFlag;
%%
%
%**************************** Store the outputs in the vector 'output' ******************************************
if nargout > 2
output.beta                 = BetaHat;
output.VaR              	= VaR;
output.y                    = y;
output.quantile             = theta;
output.Dates                = yDates;
output.Hit              	= Hit;
output.HitPercentage        = HitPercentage;
output.RQ					= RQ;
output.ExitFlag             = EXITFLAG;
output.HitPercentage      	= HitPercentage;
output.stdErr               = standardErrors;
output.coeffPvalue			= coeffPvalue;
output.initialConditions    = InitialCond;
output.Regressor            = Regressor;
end
end
if display
    columnNames = {'Coeff','StdErr','Prob'};
    if model == 1
        rowNames = {'Intercept';'LaggedBeta';'Beta';'Vol'};
    else
        rowNames = {'Intercept';'LaggedBeta';'Beta(+)';'Beta(-)';'Vol'};
    end
    TableEst = table(BetaHat,standardErrors,coeffPvalue,'RowNames',rowNames,'VariableNames',columnNames);
    disp(TableEst)
end
rng('default')
end

%%
%-------------------LOCAL FUNCTION--------------------------%
% RQobjectiveFunction to optimize the coefficients
function output = RQobjectiveFunction(BETA, OUT, MODEL, y,X, THETA, empiricalQuantile)                                                                                                       
% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
% *                                                                                                                                      *
% * By SIMONE MANGANELLI, European Central Bank, Frankfurt am Main.                                                                      *
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
%
%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1
   VaR = SAVloop(BETA,y, X, empiricalQuantile); 
%********************************************************************************************
% Model 2: Asymmetric Slope.
else   
   VaR = ASloop(BETA, y, X, empiricalQuantile);
end
Hit = THETA - (y <= VaR);
% Compute the Regression Quantile criterion.
%
RQ  = Hit'*(y - VaR);

if isinf(RQ) || (RQ ~= RQ) || ~isreal(RQ)
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

%----------------------------------------------------------------
% SAV process function

function q = SAVloop(BETA, y, X, empiricalQuantile)

% Compute the quantile time series for the Symmetric Absolute Value model,
% given the vector of returns y and the vector of parameters BETA.
%%
T = length(y);
q = zeros(T,1); q(1) = empiricalQuantile;
for t = 2:T
    q(t) = BETA(1) + BETA(2) * q(t-1) + BETA(3) * abs(y(t-1)) + BETA(4) * X(t-1);
end
end

%----------------------------------------------------------------

% AS process function
function q = ASloop(beta, y, X, empiricalQuantile)
%%
% Compute the quantile time series for the Asymmetric Absolute Value model,
% given the vector of returns y and the vector of parameters BETA.
%%
T = length(y);
q = zeros(T,1); q(1) = empiricalQuantile;
for t = 2:T
    q(t) = beta(1) + beta(2) * q(t-1) + beta(3) * abs(y(t-1)) * (y(t-1)>0) -  beta(4) * abs(y(t-1)) * (y(t-1)<=0) + beta(5) * X(t-1);
end
end

%------------------------------------------------------------------
% VarCov function
function [VCmatrix, D, gradient, BANDWIDTH] = VarianceCovariance(BETA, MODEL, T, y, X, THETA, VaR)

% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
% *                                                                                                                                      *
% * By SIMONE MANGANELLI, European Central Bank, Frankfurt am Main.                                                                      *
% * Created on 15 September, 2000. Last modified 20 February 2002.                                                                       *
% *                                                                                                                                      *
% ****************************************************************************************************************************************
% 
% 
% Compute the variance-covariance matrix of the estimated parameters using the formulae of theorems 2 and 3.
%
%*****************************************************************************************


% Compute the quantile residuals.
residuals = y - VaR;

%Bandwidth (old value ==1)
% Bandwidth of Koenker (2005) - see http://privatewww.essex.ac.uk/~jmcss/JM_JSS.pdf
%kk = median(abs(residuals-median(residuals)));
%hh = T^(-1/3)*(norminv(1-0.05/2))^(2/3)*((1.5*(normpdf(norminv(THETA)))^2)/(2*(norminv(THETA))^2+1))^(1/3);
%BANDWIDTH = kk*(norminv(THETA+hh)-norminv(THETA-hh));%c=1;
SortedRes = sort(abs(residuals));
if THETA == 0.01
    k = 40;BANDWIDTH = SortedRes(k);
elseif THETA == 0.05
    k = 60;BANDWIDTH = SortedRes(k);
elseif THETA == 0.025
    k = 50;BANDWIDTH = SortedRes(k);
else
    kk = median(abs(residuals-median(residuals)));
    hh = T^(-1/3)*(norminv(1-0.05/2))^(2/3)*((1.5*(normpdf(norminv(THETA)))^2)/(2*(norminv(THETA))^2+1))^(1/3);
    BANDWIDTH = kk*(norminv(THETA+hh)-norminv(THETA-hh));%c=1;
end
t=0;


% Initialize matrices.
derivative1 = zeros(T,1);
derivative2 = zeros(T,1);
derivative3 = zeros(T,1);
derivative4 = zeros(T,1);
derivative5 = zeros(T,1);

D = zeros(size(BETA,1));
A = D;

%
%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1
        gradient = zeros(T,4);
   for i = 2:T
       
       % VaR(i) = BETA(1) + BETA(2) * VaR(i-1) + BETA(3) * abs(y(i-1));
       derivative1(i) = 1 + BETA(2) * derivative1(i-1);
       derivative2(i) = VaR(i-1) + BETA(2) * derivative2(i-1);
       derivative3(i) = BETA(2) * derivative3(i-1) + abs(y(i-1));
       derivative4(i) = BETA(3) * derivative4(i-1) + X(i-1);
       
       gradient(i,:) = [derivative1(i), derivative2(i), derivative3(i), derivative4(i)];
      
       A = A + gradient(i,:)'*gradient(i,:);
       
       if abs(residuals(i)) <= BANDWIDTH
           t=t+1;
           D = D + gradient(i,:)'*gradient(i,:);
       end
   end

%
%********************************************************************************************
% Model 2: Asymmetric Slope.
%
else
        gradient = zeros(T,5);
    for i = 2:T

        % VaR(i) = BETA(1) + BETA(2) * VaR(i-1) + BETA(3) * y(i-1) * (y(i-1)>0) - BETA(4) * y(i-1) * (y(i-1)<0);
        derivative1(i) = 1 + BETA(2)*derivative1(i-1);
        derivative2(i) = VaR(i-1) + BETA(2)*derivative2(i-1);
        derivative3(i) = BETA(3)*derivative3(i-1) + y(i-1)*(y(i-1)>0);
        derivative4(i) = BETA(4)*derivative4(i-1) - y(i-1)*(y(i-1)<0);
        derivative5(i) = BETA(5)*derivative5(i-1) + X(i-1);
        
        gradient(i,:) = [derivative1(i), derivative2(i), derivative3(i), derivative4(i), derivative5(i)];  
        
        A = A + gradient(i,:)'*gradient(i,:);
        
        if abs(residuals(i)) <= BANDWIDTH
            t=t+1;
            D = D + gradient(i,:)'*gradient(i,:);
        end
    end
end


tStdErr=t;   % Check the k-NN bandwidth.
A = A/T;
D = D/(2*BANDWIDTH*T);

VCmatrix = THETA * (1-THETA) * inv(D) * A * inv(D) / T;
end

%------------------------------------------------------------------
% Function to mix the data
function Output = MixedFreqQuantile(DataY,DataYdate,DataX,DataXdate,xlag,period,Ovlap)

nobs = size(DataY,1); 
nobsShort = nobs-xlag-period+1;
DataYlowfreq = zeros(nobsShort,1);
DataYDateLow = zeros(nobsShort,1);
for t = xlag+1 : nobs-period+1
    DataYlowfreq(t-xlag,1) = sum(DataY(t:t+period-1));
    DataYDateLow(t-xlag,1) = DataYdate(t);
end
if ~Ovlap
    DataYlowfreq = DataYlowfreq(1:period:end,:);
    DataYDateLow = DataYDateLow(1:period:end,:);
end
% Set the start date and end date according to xlag, period and ylag
minDateY = DataYDateLow(1);
minDateX = DataXdate(xlag+1);
if minDateY > minDateX
    estStart = minDateY;
else
    estStart = minDateX;
end
maxDateY = DataYDateLow(end);
maxDateX = DataXdate(end);
if maxDateY > maxDateX
    estEnd = maxDateX;
else
    estEnd = maxDateY;
end

% Construct Y data
tol = 1e-10;
locStart = find(DataYDateLow >= estStart-tol, 1);
locEnd = find(DataYDateLow >= estEnd-tol, 1);
EstY = DataYlowfreq(locStart:locEnd);
EstYdate = DataYDateLow(locStart:locEnd);

nobsEst = size(EstY,1);
% Construct lagged X data
EstX = zeros(nobsEst,xlag);
EstXdate = zeros(nobsEst,xlag);
for t = 1:nobsEst
    loc = find(DataXdate >= EstYdate(t)-tol, 1);
    if isempty(loc)
        loc = length(DataXdate);
    end
    
    if loc > size(DataX,1)        
        nobsEst = t - 1;
        EstY = EstY(1:nobsEst,:);
        EstYdate = EstYdate(1:nobsEst,:);
        EstX = EstX(1:nobsEst,:);
        EstXdate = EstXdate(1:nobsEst,:);
        maxDate = EstYdate(end);
        warning('MixFreqData:EstEndOutOfBound',...
            'Horizon is a large negative number. Observations are further truncated to %s',datestr(maxDate))
        break
    else        
        EstX(t,:) = DataX(loc-1:-1:loc-xlag);
        EstXdate(t,:) = DataXdate(loc-1:-1:loc-xlag);
    end    
end

Output = struct('EstY',EstY,'EstYdate',EstYdate,'EstX',EstX,'EstXdate',EstXdate,...
    'EstStart',estStart,'EstEnd',estEnd);
end