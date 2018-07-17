function [estParams,condQuantile,output] = MidasQuantileAS(y,varargin)
%MidasQuantile: MIDAS quantile regression
%
% Syntax:
%
%   [estParams,condQuantile] = MidasQuantile(y)
%   [estParams,condQuantile,yLowFreq,xHighFreq,yDates] = MidasQuantile(y, name,value,...)
%
% Description:
%
%  Suppose that we have a return series y(t), t = 1,...,T.
%  MIDAS quantile regression estimates the conditional quantile of n-period
%  returns (obtained by aggregating y(t),...,y(t+n)). The conditioning
%  variable (predictor) is sampled at high frequency with MIDAS weights.
%  The default predictor is |y(t)|, as "absolute returns successfully 
%  capture time variation in the conditional distribution of returns".
%
%  Refer to Ghysels, Plazzi and Valkanov (2016) for model specification.
%
% Input Arguments:
%
%   y           T-by-1 observation data
%
% Optional Input Name/Value Pairs:
%
%  'Quantile'  A scalar between zero and one that specifies the level
%              (i.e., alpha) of quantile. The default is 0.05.
%
%  'X'         T-by-1 high frequency conditioning variable (predictor). 
%              This variable must have the same length as y, and only one
%              predictor is supported. By default, it is |y|, as "absolute
%              returns successfully capture time variation in the 
%              conditional distribution of returns".
%
%  'Period'    A scalar integer that specifies the aggregation periodicity.
%              y will be aggregated so as to formulate n-period returns.
%              How many days in a week/month/quarter/year?
%              The default is 22 (as in a day-month aggregation)
%
%  'NumLags'   A scalar integer that specifies the number of lags for the
%              high frequency predictor, to which MIDAS weights is
%              assigned. The default is 250.
%
%  'Dates'     T-by-1 vector or cell array for the dates of y. This is
%              for book-keeping purpose and does not affect estimation.
%              The default is 1:length(y).
%
%  'Search'    A logical value that indicates numerical minimization
%              via pattern search in the Global Optimization Toolbox.
%              If not available, it resorts to fminsearch in base MATLAB.
%              The default is false (and will use gradient-based methods
%              under smoothed objective functions)
%
%  'Options'   The options for numerical optimization. 
%              The default is the FMINCON default choice.
%
%  'Gradient'  A logical value that indicates analytic gradients.
%              The default is false.
%
%  'Bootstrap' A character vector that specifies bootstrap standard error
%              method: 'Residual' (Default) or 'XY'.
%
%  'Params'    Parameter values for [intercept;slope;k].
%              In that case, the program skips estimation, and just infers
%              conditional quantiles based on the specified parameters.
%              The default is empty (need parameter estimation).
%
% Output Arguments:
%
%   estParams   Estimated parameters for [intercept;slope;k],
%               where intercept and slope are the coefficients of the
%               quantile regression, and k is the parameter in the
%               MIDAS Beta polynomial
%
%   condQuantile R-by-1 conditional quantile. This is the fitted value of
%                the right-hand-side of the quantile regression.
%                R = T - Period - NumLags + 1.
%
%   yLowFreq     R-by-1 n-period returns obtained from overlapping
%                aggregation of y. This is the left-hand-side of the
%                quantile regression.
%
%   xHighFreq    R-by-NumLags high-frequency predictor data used by the
%                quantile regression
%
%   yDates       R-by-1 serial dates for the output variables.
%
%
% Notes:
%
%
% o If numerical optimization works poorly, try 'Gradient' = 1.
%   The codes will be slower for MATLAB versions earlier than R2015b.
%
% Reference:
%
% Ghysels, E., Plazzi, A. and Valkanov, R. (2016) Why Invest in Emerging
% Market? The Role of Conditional Return Asymmetry. Journal of Finance,
% forthcoming.

%%
% Parse inputs and set defaults
quantileDefault = 0.05;
periodDefault = 1;
nlagDefault = 100;
callerName = 'MidasQuantileAS';
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'scalar','>',0,'<',1},callerName));
addParameter(parseObj,'X',[],@(x)validateattributes(x,{'numeric'},{'2d'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'NumLags',nlagDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'xDates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'Ovlap',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'DoParallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Cores',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'numInitials',5,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'numInitialsRand',20000,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Beta2Para',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'GetSe',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Display',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Search',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Options',[],@(x)validateattributes(x,{},{},callerName));
addParameter(parseObj,'Params',[],@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'Constrained',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'startPars',[]);

parse(parseObj,varargin{:});
q = parseObj.Results.Quantile;
Regressor = parseObj.Results.X;
period = parseObj.Results.Period;
nlag = parseObj.Results.NumLags;
yDates = parseObj.Results.Dates;
xDates = parseObj.Results.xDates;
ovlap = parseObj.Results.Ovlap;
doparallel = parseObj.Results.DoParallel;
cores = parseObj.Results.Cores;
numInitials = parseObj.Results.numInitials;
numInitialsRand = parseObj.Results.numInitialsRand;
beta2para = parseObj.Results.Beta2Para;
searchFlag = parseObj.Results.Search;
options = parseObj.Results.Options;
getse = parseObj.Results.GetSe;
display = parseObj.Results.Display;
estParams = parseObj.Results.Params;
constrained = parseObj.Results.Constrained;
startPars = parseObj.Results.startPars;
% Replace missing values by the sample average
y = y(:);
y(isnan(y)) = nanmean(y);
nobs = length(y);
% Load dates
if iscell(yDates)
    yDates = datenum(yDates);    
end
if numel(yDates) ~= nobs
    error('Length of Dates must equal the number of observations.')
end

if iscell(xDates)
    xDates = datenum(xDates);    
end

% Load the conditioning variable (predictor)
if isempty(Regressor)
    Regressor = abs(y);
    if isempty(xDates)
        xDates = yDates;
    end
else
    if numel(Regressor) ~= numel(y)
        error('Conditioning variable (predictor) must be a vector of the same length as y.')
    end
    if isempty(xDates)
        error('Regressors dates need to be supplied')
    end
    Regressor = Regressor(:);
    Regressor(isnan(Regressor)) = nanmean(Regressor);
end

if numel(xDates) ~= nobs
    error('Length of xDates must equal the number of observations.')
end

if doparallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local',cores);
    end
end

%%

% Prepare data for the LHS and RHS of the quantile regression
% LHS: n-period returns by aggregating y(t),...,y(t+period-1)
% RHS: many lagged returns by extracting y(t-1),...,y(t-nlag)

MixedRet = MixedFreqQuantile(y,yDates,y,yDates,nlag,period,ovlap);
yHighOri = MixedRet.EstX;

MixedData = MixedFreqQuantile(y,yDates,Regressor,xDates,nlag,period,ovlap);
yLowFreq = MixedData.EstY;
xHighFreq = MixedData.EstX;
yDates = MixedData.EstYdate;
xDates = MixedData.EstXdate;
nobsEst = size(yLowFreq,1);

%%
% In case of known parameters, just compute conditional quantile and exit.
if ~isempty(estParams)
    [~,condQuantile] = objFun(estParams,yLowFreq,yHighOri,xHighFreq,q,beta2para);
    Hit = q - (yLowFreq <= condQuantile);
    HitPercentage    = mean(Hit(1:nobsEst) + q) * 100;
    if nargout > 1
    output.estParams = estParams;
    output.CondQ = condQuantile;
    output.yLowFreq = yLowFreq;
    output.Hit = Hit; 
    output.HitPercentage = HitPercentage;
    output.yDates = yDates;
    end
    return
end
%%
% Optimazation procedues
if isempty(startPars)
fprintf('Finding the initial Betas... \n');
betaIni = GetIniParams(yLowFreq, yHighOri, xHighFreq, q, nlag, beta2para,numInitialsRand,numInitials,doparallel);
else
betaIni = startPars';
end
fprintf('Optimizing parameters.... \n');
% Optimization options
MaxFunEvals = 2000; % Increase in case the model is hard to converge
MaxIter = 2000;

if isempty(options)
    if ~searchFlag
        options = optimoptions('fmincon','GradObj','on','DerivativeCheck','on','FinDiffType','central','Algorithm','interior-point','Display','notify-detailed');
    else 
        options = optimset('Display','off','MaxFunEvals',MaxFunEvals,'MaxIter',MaxIter, 'TolFun', 1e-8, 'TolX', 1e-8);
    end        
end

optionsUnc = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxFunEvals',MaxFunEvals,'MaxIter',MaxIter);

optionCon = optimoptions('fmincon', 'MaxFunEvals', MaxFunEvals, ...
                    'display', 'off', 'MaxIter', MaxIter);
   tol = 1e-7;
    if ~beta2para
    lb = [-Inf;-Inf;-Inf;1+tol];
    ub = [Inf;Inf;Inf;200-tol];
    else
    lb = [-Inf;-Inf;-Inf;1+tol;1+tol];
    ub = [Inf;Inf;Inf;200-tol;200-tol];
    end

REP = 5; %Number of time the argorithm will be restarted.
% Numeric minimization
estParams = zeros(size(betaIni));
fval = zeros(size(betaIni,1),1);
exitFlag = zeros(size(betaIni,1),1);
if doparallel
parfor i = 1:size(betaIni,1)  
    if ~searchFlag 
    % Minimize non-differentiable function by Newton method, analytic grad
        [estParams(i,:),fval(i,1),exitFlag(i,1)] = fmincon(@(params) objFunGrad(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),betaIni(i,:),[],[],[],[],lb,ub,[],options);
    else
        [estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearch(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),betaIni(i,:),options);         
        for ii = 1:REP
            try
            [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminunc(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),optionsUnc);
            catch
            warning('fminunc does not work. Move on to the fminsearch.');    
            end
            [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearch(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),options);
            if constrained
                [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearchbnd(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),lb,ub,options);
            end            
            if exitFlag(i,1) == 1
                break
            end
        end
    end
end
else
for i = 1:size(betaIni,1)  
    if ~searchFlag
    % Minimize non-differentiable function by Newton method, analytic grad
        [estParams(i,:),fval(i,1),exitFlag(i,1)] = fmincon(@(params) objFunGrad(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),betaIni(i,:),[],[],[],[],lb,ub,[],options);
    else
        [estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearch(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),betaIni(i,:),options);         
        for ii = 1:REP
            try
            [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminunc(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),optionsUnc);
            catch
            warning('fminunc does not work. Move on to the fminsearch.');    
            end
            [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearch(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),options);
            if constrained
                [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearchbnd(@(params) objFun(params,yLowFreq,yHighOri,xHighFreq,q,beta2para),estParams(i,:),lb,ub,options);
            end            
            if exitFlag(i,1) == 1
                break
            end
        end
    end
end
end
SortedFval = sortrows([fval,exitFlag,estParams],1);
estParams = SortedFval(1,3:size(SortedFval,2))';
exitFlag = SortedFval(1,2);
[fval,condQuantile] = objFun(estParams,yLowFreq,yHighOri,xHighFreq,q,beta2para);
%%
% Bootstrap standard errors
if getse
fprintf('Getting standard errors... \n');
nsim = 200;
resid = (yLowFreq - condQuantile);
paramSim = zeros(length(estParams),nsim);
if doparallel
parfor r = 1:nsim
    ind = randi(nobsEst,[nobsEst,1]);
    residSim = resid(ind);
    %xHighFreqSim = xHighFreq(ind,:);
    %yHighOriSim = yHighOri(ind,:);
    %[yLowFreqSim,~] = GetSim(estParams,yHighOri,xHighFreq,beta2para,residSim);
    yLowFreqSim = condQuantile + residSim; 
    paramSim(:,r) = fminsearch(@(params) objFun(params,yLowFreqSim,yHighOri,xHighFreqSim,q,beta2para),estParams,options);
end
else
for r = 1:nsim
    ind = randi(nobsEst,[nobsEst,1]);
    residSim = resid(ind); 
    %xHighFreqSim = xHighFreq(ind,:);
    %yHighOriSim = yHighOri(ind,:);
    %[yLowFreqSim,~] = GetSim(estParams,yHighOri,xHighFreq,beta2para,residSim);
    yLowFreqSim = condQuantile + residSim; 
    paramSim(:,r) = fminsearch(@(params) objFun(params,yLowFreqSim,yHighOri,xHighFreq,q,beta2para),estParams,options);
end
end
se = std(paramSim,0,2);
%zstat = estParams ./ se;
%pval = 0.5 * erfc(0.7071 * abs(zstat)) * 2;
%pval(pval<1e-6) = 0;
if beta2para
    hypothesis = [0;0;0;1;1];
else
    hypothesis = [0;0;0;1];
end
% Hypothesis that the betaLags is not equal to 1, which mean equally
% weighted (i.e., no point of using MIDAS lags);
meanParamSim = repmat(mean(paramSim,2),1,nsim);
pval =  mean(abs(paramSim - meanParamSim + hypothesis) > repmat(abs(estParams),1,nsim),2);
else
se = nan(length(estParams),1); 
pval = nan(length(estParams),1);
end
%%
% Display the estimation results
columnNames = {'Coeff','StdErr','Prob'};
if beta2para
  rowNames = {'Intercept';'Slope(+)';'Slope(-)';'k1';'k2'};
else
  rowNames = {'Intercept';'Slope(+)';'Slope(-)';'k2'};
end
    TableEst = table(estParams,se,pval,'RowNames',rowNames,'VariableNames',columnNames);
if display
if ~searchFlag && ~autodiffFlag
    fprintf('Method: Asymmetric loss function minimization\n');
elseif ~searchFlag && autodiffFlag
    fprintf('Method: Asymmetric loss function minimization, Analytic gradient Newton iterations\n');
else
    if exist('patternsearch','file') ~= 0        
        fprintf('Method: Asymmetric loss function minimization, Pattern search\n');
    else        
        fprintf('Method: Asymmetric loss function minimization, Nelder-Mead search\n');
    end 
end
fprintf('Sample size:                 %d\n',nobs);
fprintf('Adjusted sample size:        %d\n',nobsEst);
fprintf('Minimized function value: %10.6g\n',fval);
disp(TableEst)
end
%%
% Get the output file
Hit = q - (yLowFreq <= condQuantile);
HitPercentage    = mean(Hit(1:nobsEst) + q) * 100;
if nargout > 1
output.estParams = estParams;
output.CondQ = condQuantile;
output.fval = fval;
output.exitFlag = exitFlag;
output.TableEst = TableEst;
output.se = se; 
output.pval = pval;
output.Hit = Hit;
output.HitPercentage = HitPercentage;
output.nobs = nobsEst;
output.yLowFreq = yLowFreq;
output.nlag = nlag;
output.xHighFreq = xHighFreq;
output.quantile = q; 
output.beta2para = beta2para;
output.horizon = period;
output.yDates = yDates; 
output.xDates = xDates;
if beta2para
output.weights = estParams(2) * midasBetaWeights(nlag,estParams(4),estParams(5));
else
output.weights = estParams(2) * midasBetaWeights(nlag,1,estParams(4));
end
end
rng('default')
end

%-------------------------------------------------------------------------
% Local function: the beta weights function
function weights = midasBetaWeights(nlag,param1,param2)
seq = linspace(eps,1-eps,nlag);
if param1 == 1    
    weights = (1-seq).^(param2-1);    
else
    weights = (1-seq).^(param2-1) .* seq.^(param1-1);    
end
weights = weights ./ nansum(weights);
end
%-------------------------------------------------------------------------
% Local function: the beta weights function CAPITAL
function weights = midasBetaWeightsCapital(nlag,param1,param2)
seq = linspace(eps,1-eps,nlag);
if param1 == 1
    weights = POWER(1-seq, param2-1);
else
    weights = TIMES( POWER(1-seq, param2-1), POWER(seq, param1-1)) ;
end
weights = RDIVIDE(weights,nansum(weights));
end
%-------------------------------------------------------------------------
% Local function: the mixing data sample function
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
%-------------------------------------------------------------------------
% Local function: the function to get initial beta estimations
function beta = GetIniParams(yLowFreq, yHighOri, xHighFreq, q, nlag, beta2para,numInitialsRand,numInitials,doparallel)
% Randomly sample second parameter of Beta polynomial  
nInitalBeta = unifrnd(1,100,[numInitialsRand,1]);
nobsEst = size(yLowFreq,1);
nParams = 4 + beta2para;
InitialParamsVec = zeros(numInitialsRand,nParams);
ols = zeros(3,numInitialsRand);

for i = 1:numInitialsRand
    X0 = [ones(nobsEst,1), xHighFreq * midasBetaWeights(nlag,1,nInitalBeta(i))'];
    ols(1:2,i) = X0\yLowFreq;
    ols(3,i) = ols(2,i);
    if beta2para
    InitialParamsVec(i,:) = [ols(:,i)',1,nInitalBeta(i)];
    else
    InitialParamsVec(i,:) = [ols(:,i)',nInitalBeta(i)];
    end
end
RQfval = zeros(numInitialsRand,1);
if doparallel
parfor i = 1:numInitialsRand
   RQfval(i) = objFun(InitialParamsVec(i,:),yLowFreq,yHighOri, xHighFreq,q, beta2para);
end
else
for i = 1:numInitialsRand
   RQfval(i) = objFun(InitialParamsVec(i,:),yLowFreq,yHighOri, xHighFreq,q,beta2para);
end
end
Results = [RQfval,InitialParamsVec];
SortedResults = sortrows(Results,1);
beta = SortedResults(1:numInitials,2:size(SortedResults,2));
end
%-------------------------------------------------------------------------
% Local function: the objective function
function [fval,condQuantile] = objFun(params,y,yHighOri,X,q,beta2para)
%function [fval,condQuantile] = objFun(params,y,yHighOri,X,q) 
% Allocate parameters
intercept = params(1);
slope1 = params(2);
slope2 = params(3);
if beta2para
k1 = params(4);
k2 = params(5);
else
k1 = 1;
k2 = params(4);
end
% Compute MIDAS weights
nlag = size(X,2);
weights = midasBetaWeights(nlag,k1,k2)';
%nobs = length(y); 
% Conditional quantile
%condQuantile = zeros(nobs,1);
%for t = 1:nobs
X_neg = X .* (yHighOri<=0);
X_pos = X .* (yHighOri>0);
condQuantile = intercept + slope1 .* (X_pos * weights) + slope2 .* (X_neg * weights);
%end
% Asymmetric loss function
loss = y - condQuantile;
fval = loss' * (q - (loss<0));
end

%-------------------------------------------------------------------------
% Local function: the objective function
function [fval,condQuantile] = objFunCapital(params,y, yHighOri,X,q,beta2para)
% Allocate parameters
intercept = params(1);
slope1 = params(2);
slope2 = params(3);
if beta2para
k1 = params(4);
k2 = params(5);
else
k1 = 1;
k2 = params(4);
end
% Compute MIDAS weights
nlag = size(X,2);
weights = midasBetaWeightsCapital(nlag,k1,k2)';

% Conditional quantile
condQuantile = zeros(nobs,1);
for t = 1:nobs
X_neg = (yHighOri(t,:)<=0);
X_pos = (yHighOri(t,:)>0);
condQuantile(t) = intercept + TIMES(slope1,MTIMES(TIMES(X(t,:),X_pos),weights)) + TIMES(slope2,MTIMES(TIMES(X(t,:),X_neg),weights));
end

% Asymmetric loss function
loss = y - condQuantile;
    % Non-differentiable loss function    
    fval = MTIMES(loss.', (q - (loss<0)));
end


%-------------------------------------------------------------------------
% Local function: Compute objective function value and gradient
function [fval,Gradient] = objFunGrad(params,y, yHighOri,X,q,beta2para)
fval = objFun(params,y, yHighOri,X,q,beta2para);
if nargout == 1    
    return
end

fun = @(params) objFunCapital(params,y, yHighOri,X,q,beta2para);
x = params;

nfunVal = numel(fval);
nparams = numel(x);
fvalVec = zeros(nfunVal,nparams);
Gradient = zeros(nfunVal,nparams);

for m = 1:nparams
    
    % One element of x carries an imaginary number
    xComplex = x;
    xComplex(m) = xComplex(m) + 1i;
    
    % The real component is the function value 
    % The imaginary component is the derivative
    complexValue = fun(xComplex);
    fvalVec(:,m) = real(complexValue);
    Gradient(:,m) = imag(complexValue);
end

if any(norm(bsxfun(@minus,fvalVec, fval))>1e-6)
    warning('Gradients might be falsely computed. Capitalize arithmetic operator and utility function names.')
end

end

%-------------------------------------------------------------------------
% Local function: Compute the yLowFreqSim

function [yLowFreqSim,CondQsim] = GetSim(params,yHighOriSim,xHighFreqSim,beta2para,ResidSim)
% Allocate the parameters
intercept = params(1);
slope1 = params(2);
slope2 = params(3);
if beta2para
k1 = params(4);
k2 = params(5);
else
k1 = 1;
k2 = params(4);
end
% Compute MIDAS weights
%nobs = size(xHighFreqSim,1);
nlag = size(xHighFreqSim,2);
weights = midasBetaWeights(nlag,k1,k2)';
% Conditional quantile
%CondQsim = zeros(nobs,1);
%for t = 1:nobs
X_neg = (yHighOriSim<=0);
X_pos = (yHighOriSim>0);
CondQsim = intercept + slope1 .* ((xHighFreqSim .* X_pos) * weights) + slope2 .* ((xHighFreqSim .* X_neg) * weights);
%end
yLowFreqSim = CondQsim + abs(CondQsim).*ResidSim;
end

%-------------------------------------------------------------------------
function B = EXP(A)
Areal = real(A);
Aimag = imag(A);
Breal = exp(Areal);
Bimag = Breal .* Aimag;
B = complex(Breal, Bimag);
end

%-------------------------------------------------------------------------
function B = LOG(A)
Areal = real(A);
Aimag = imag(A);
Breal = log(Areal);
Bimag = Aimag ./ Areal;
B = complex(Breal, Bimag);
end


%-------------------------------------------------------------------------
function C = MTIMES(A,B)
C = A * B + imag(A) * imag(B);
end


%-------------------------------------------------------------------------
function C = POWER(A,B)
if isscalar(B) && (B == 3)
    C = TIMES(TIMES(A, A), A);
elseif isscalar(B) && (B == 2)
    C = TIMES(A, A);
elseif isscalar(B) && (B == 1)
    C = A;
elseif isscalar(B) && (B == 0.5)
    C = SQRT(A);
elseif isscalar(B) && (B == 0)
    C = complex(ones(size(A)),0);
elseif isscalar(B) && (B == -0.5)
    C = RDIVIDE(1, SQRT(A));
elseif isscalar(B) && (B == -1)
    C = RDIVIDE(1, A);
elseif isscalar(B) && (B == -2)
    C = RDIVIDE(1, TIMES(A,A));
elseif all(A(:)>0)
    C = EXP(TIMES(B, LOG(A)));
else
    % This is not necessarily correct
    Creal = real(A).^real(B);
    Cimag = imag(exp(B.*log(abs(A))));
    C = complex(Creal,Cimag);
end
end


%-------------------------------------------------------------------------
function C = RDIVIDE(A,B)
Areal = real(A);
Aimag = imag(A);
Breal = real(B);
Bimag = imag(B);
Creal = Areal ./ Breal;
Cimag = (Aimag .* Breal - Areal .* Bimag) ./ (Breal.*Breal);
C = complex(Creal, Cimag);
end


%-------------------------------------------------------------------------
function C = TIMES(A,B)
C = A .* B + imag(A) .* imag(B);
end
