    function [estParams,CondQ,CondES,output] = VarEs_CAViaR1(y,varargin)
%VARES Summary of this function goes here
%   Detailed explanation goes here
quantileDefault = 0.05;
periodDefault = 1;
nlagDefault = 0;
callerName = 'VarEs_CAViaR1';
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'scalar','>',0,'<',1},callerName));
addParameter(parseObj,'X',[],@(x)validateattributes(x,{'numeric'},{'2d'},callerName));
addParameter(parseObj,'Model',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'NumLags',nlagDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'xDates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'DoParallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Cores',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'numInitials',10,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'numInitialsRand',100000,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'GetSe',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Display',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Options',[],@(x)validateattributes(x,{},{},callerName));
addParameter(parseObj,'Params',[],@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'Constrained',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'EmpQuant',[],@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'mu0',[],@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'startPars',[]);

parse(parseObj,varargin{:});
theta = parseObj.Results.Quantile;
Regressor = parseObj.Results.X;
model = parseObj.Results.Model;
yDates = parseObj.Results.Dates;
period = parseObj.Results.Period;
nlag = parseObj.Results.NumLags;
doparallel = parseObj.Results.DoParallel;
cores = parseObj.Results.Cores;
numInitials = parseObj.Results.numInitials;
numInitialsRand = parseObj.Results.numInitialsRand;
options = parseObj.Results.Options;
getse = parseObj.Results.GetSe;
display = parseObj.Results.Display;
BetaHat = parseObj.Results.Params;
xDates = parseObj.Results.xDates;
constrained = parseObj.Results.Constrained;
empiricalQuantile = parseObj.Results.EmpQuant;
mu0 = parseObj.Results.mu0;
startPars = parseObj.Results.startPars;

if ~ismember(model,[1,2])
    disp(' ')
    disp('*******************************************************************')
    disp('ERROR! You need to select one of the following models:')
    disp('Model=1: Symmetric Absolute Value')
    disp('Model=2: Asymmetric Slope')
end

% Replace missing values by the sample average
if isempty(mu0)
    mu0 = 0; %Empirical mean to adjust the variable
end

y = y(:);
y(isnan(y)) = nanmean(y);
nobs = length(y);
y = y - mu0;

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

if ~isempty(Regressor) && isempty(xDates)
error('Regressor needs to be provided with Dates.')
end

if doparallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local',cores);
    end
end

% Mix data according to the return horizon and nlag 
ovlap = false; 
MixedData = MixedFreqQuantile(y,yDates,y,yDates,nlag,period,ovlap);
y = MixedData.EstY;
yDates = MixedData.EstYdate;
nobs = size(y,1);
% Load the conditioning variable (predictor): At the moment, just allow
% Regressor to be daily variance-type value
if ~isempty(Regressor)
   MixedVar = MixedFreqQuantile(Regressor,xDates,Regressor,xDates,nlag,period,ovlap);
   Regressor = sqrt(MixedVar.EstY);
   xDates = MixedData.EstYdate;
end 
%%
% Get the empirical quantile
if isempty(empiricalQuantile)
   if period == 1
       InitialEmp = 300;
   elseif period == 5
       InitialEmp = 100;
   else
       InitialEmp = 50;
   end
   ysort          = sortrows(y(1:InitialEmp), 1); 
   %empiricalQuantile = ysort(round(InitialEmp*theta));
   empiricalQuantile = quantile(ysort,theta);
end
%%
% In case of known parameters, just compute conditional quantile/ES and exit.
if ~isempty(BetaHat)
    [~,CondQ,CondES] = ALdist1(BetaHat,y,Regressor,theta,model,empiricalQuantile);
    estParams = BetaHat;
    Hit = theta - (y <= CondQ);
    HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
    CondQ = CondQ + period*mu0;
    CondES = CondES + period*mu0;
    y = y + period * mu0;
    if nargout > 3
    output.estParams = estParams;
    output.quantile = theta;
    output.y = y;
    output.Hit = Hit;
    output.HitPercentage = HitPercentage;
    output.VaR = CondQ;
    output.ES = CondES;
    output.Dates = yDates;
    output.Regressor = Regressor;
    output.xDates = xDates;
    end
    return
end
%%
% Optimization
  if isempty(startPars)
% First get the parameter estimates of the quantile regression
    fprintf('Estimating univariate CAViaR estimate to get initial parameters\n');
    if ~isempty(Regressor)
    QuantEst = CAViaR_X(y,'Dates',yDates,'X',Regressor','xDates',xDates,'Model',model,...
        'Display',false,'GetSe',false,'DoParallel',doparallel,'Cores',cores,...
        'DoFull',true,'Quantile',theta,'Constrained',constrained,'Period',period,...
        'NumLags',nlag);
    else
    QuantEst = CAViaR_Uni(y,'Dates',yDates,'Model',model,...
        'Display',false,'GetSe',false,'DoParallel',doparallel,'Cores',cores,...
        'DoFull',true,'Quantile',theta,'Constrained',constrained,'Period',period,...
        'NumLags',nlag);
    end
% Get the initial parameters for the AL distribution
fprintf('Estimating initial parameters for VaREs\n');
betaIni = IniParAL1(QuantEst,y, Regressor, theta, empiricalQuantile,numInitialsRand,numInitials,model,doparallel);
  else
      betaIni = startPars';
  end
% Optimization options
MaxFunEvals = 3000; 
MaxIter = 3000;
fprintf('Optimizing parameters.... \n');

if isempty(options)
   options = optimset('LargeScale', 'off', 'HessUpdate','dfp','Display','off',...
       'MaxFunEvals',MaxFunEvals,'MaxIter',MaxIter, 'TolFun', 1e-8, 'TolX', 1e-8);
end

optionsUnc = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton',...
    'MaxFunEvals',MaxFunEvals,'MaxIter',MaxIter);

optionCon = optimoptions('fmincon', 'MaxFunEvals', MaxFunEvals, ...
     'display', 'off', 'MaxIter', MaxIter);
 
   tol = 1e-7;
   if ~isempty(Regressor)
      if model == 1
      lb = [-Inf;0+tol;-Inf;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf;Inf];
      else
      lb = [-Inf;0+tol;-Inf;-Inf;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf;Inf;Inf];
      end 
   else
      if model == 1
      lb = [-Inf;0+tol;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf];
      else
      lb = [-Inf;0+tol;-Inf;-Inf;-Inf];
      ub = [Inf;Inf;Inf;Inf;Inf];
      end
   end
   
% Numeric minimization
REP = 15;
estParams = zeros(size(betaIni));
fval = zeros(size(betaIni,1),1);
exitFlag = zeros(size(betaIni,1),1);
if doparallel
parfor i = 1:size(betaIni,1)
    [estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearch(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),betaIni(i,:),options);        
    %[estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearchbnd(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),betaIni(i,:),lb,ub,options);        
    for ii = 1:REP
    try
    [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminunc(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),optionsUnc);
    catch
    warning('fminunc does not work. Move on to the fminsearch.');
    end   
        [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearch(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),options);
    if constrained
        %[estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearchbnd(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),lb,ub,options);
        if sum(estParams(i,:)' > lb) + sum(estParams(i,:)' < ub) ~= (2*length(estParams(i,:))) % If fminsearch violate the bounds, redo the optimization with fmincon
           [estParams(i,:), fval(i,1), exitFlag(i,1)] = fmincon(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),[],[],[],[],lb,ub,[],optionCon);
       end
    end
       if exitFlag(i,1) == 1
          break
       end
    end
end
else
for i = 1:size(betaIni,1)
    [estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearch(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),betaIni(i,:),options);        
    %[estParams(i,:),fval(i,1),exitFlag(i,1)] = fminsearchbnd(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),betaIni(i,:),lb,ub,options);
    for ii = 1:REP
    try
    [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminunc(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),optionsUnc);
    catch
    warning('fminunc does work. Move on to the fminsearch.');
    end
      [estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearch(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),options);
    if constrained
       %[estParams(i,:),fval(i,1),exitFlag(i,1)]  = fminsearchbnd(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),lb,ub,options);
       if sum(estParams(i,:)' > lb) + sum(estParams(i,:)' < ub) ~= (2*length(estParams(i,:))) % If fminsearch violate the bounds, redo the optimization with fmincon
          [estParams(i,:), fval(i,1), exitFlag(i,1)] = fmincon(@(params) ALdist1(params,y,Regressor,theta,model,empiricalQuantile),estParams(i,:),[],[],[],[],lb,ub,[],optionCon);
       end
    end
       if exitFlag(i,1) == 1
          break
       end
    end
end    
end
SortedFval = sortrows([fval,exitFlag,estParams],1);
estParams = SortedFval(1,3:size(SortedFval,2))';
fval = SortedFval(1,1);
exitFlag = SortedFval(1,2);
[~,CondQ,CondES] = ALdist1(estParams,y,Regressor,theta,model,empiricalQuantile);
%%
%Get standard errors using the simulation approach
if getse
fprintf('Calculating standard errors\n');
nsim = 100;
resid = (y - CondQ)./abs(CondQ);
paramSim = zeros(length(estParams),nsim);
FirstY = y(1); 
FirstCondQ = CondQ(1); 
FirstCondES = CondES(1);
if doparallel
parfor r = 1:nsim
    ind = randi(nobs,[nobs,1]);
    residSim = resid(ind);
    if ~isempty(Regressor)
    XSim = Regressor(ind);
    else
    XSim = [];
    end
    [ySim,~,~] = GetSim(estParams,model,FirstY,XSim,FirstCondQ,FirstCondES,residSim);
    paramSim(:,r) = fminsearch(@(params) ALdist1(params,ySim,XSim,theta,model,empiricalQuantile),estParams,options);
end
else
for r = 1:nsim
    ind = randi(nobs,[nobs,1]);
    residSim = resid(ind);
    if ~isempty(Regressor)
    XSim = Regressor(ind);
    else
    XSim = [];
    end
    [ySim,~,~] = GetSim(estParams,model,FirstY,XSim,FirstCondQ,FirstCondES,residSim);
    paramSim(:,r) = fminsearch(@(params) ALdist1(params,ySim,XSim,theta,model,empiricalQuantile),estParams,options);
end    
end
se = std(paramSim,0,2);
zstat = estParams ./ se;
%pval = 0.5 * erfc(0.7071 * abs(zstat)) * 2;
%pval(pval<1e-6) = 0;
meanParamSim = repmat(mean(paramSim,2),1,nsim);
pval =  mean(abs(paramSim - meanParamSim) > repmat(abs(estParams),1,nsim),2);
else
se = nan(length(estParams),1); 
pval = nan(length(estParams),1);
zstat = nan(length(estParams),1); 
end
%%
% Get all estimation table
columnNames = {'Coeff','StdErr','tStat','Prob'};
if ~isempty(Regressor)
if model == 1
   rowNames = {'Intercept';'LaggedBeta';'Beta';'Regressor';'phi'};
else
   rowNames = {'Intercept';'LaggedBeta';'Beta(+)';'Beta(-)';'Regressor';'phi'};
end
else
if model == 1
   rowNames = {'Intercept';'LaggedBeta';'Beta';'phi'};
else
   rowNames = {'Intercept';'LaggedBeta';'Beta(+)';'Beta(-)';'phi'};
end
end
    TableEst = table(estParams,se,zstat,pval,'RowNames',rowNames,'VariableNames',columnNames);
%%
% Get the estimation output
Hit = theta - (y <= CondQ);
HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
CondQ = CondQ + period*mu0;
CondES = CondES + period*mu0;
y = y + period * mu0;
if nargout > 3
output.estParams = estParams;
output.se = se;
output.quantile = theta;
output.pval = pval;
output.y = y;
output.Hit = Hit;
output.HitPercentage = HitPercentage;
output.VaR = CondQ;
output.ES = CondES;
output.Dates = yDates;
output.TableEst = TableEst;
output.exitFlag = exitFlag;
output.fval = fval;
output.Regressor = Regressor;
output.xDates = xDates;
if isempty(startPars)
output.QuantEst = QuantEst;
end
end
%%
if display      
fprintf('Method: Asymmetric loss function minimization, Nelder-Mead search\n');
if exitFlag == 1
    fprintf('Optimization succeeded \n');
else
    fprintf('Optimization failed\n');
end
fprintf('Sample size:                 %d\n',nobs);
fprintf('Minimized function value: %10.6g\n',fval);
disp(TableEst);
end
end

%%
% Local Function
%-------------------------------------------------------
% Local function for SAV CaviaR
function q = SAVloop(BETA, y, X, empiricalQuantile)
%%
% Compute the quantile time series for the Symmetric Absolute Value model,
% given the vector of returns y and the vector of parameters BETA.
%%
T = length(y);
q = zeros(T,1); q(1) = empiricalQuantile;
if ~isempty(X)
for t = 2:T
    q(t) = BETA(1) + BETA(2) * q(t-1) + BETA(3) * abs(y(t-1)) + BETA(4) * X(t-1);
end
else
for t = 2:T
    q(t) = BETA(1) + BETA(2) * q(t-1) + BETA(3) * abs(y(t-1));
end
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
if ~isempty(X)
for t = 2:T
    q(t) = beta(1) + beta(2) * q(t-1) + beta(3) * y(t-1) * (y(t-1)>0) -  beta(4) * y(t-1) * (y(t-1)<=0) + beta(5) * X(t-1);
end
else
for t = 2:T
    q(t) = beta(1) + beta(2) * q(t-1) + beta(3) * y(t-1) * (y(t-1)>0) -  beta(4) * y(t-1) * (y(t-1)<=0);
end
end
end

%------------------------------------------------------------------------
% function to get initial estimation of the beta
function beta = IniParAL1(QuantEst,y, Regressor, theta, empiricalQuantile,numInitialsRand,numInitials,model,doparallel)
% Randomly sample the ES parameter. With the application at hand, the ES
% parameter seems to be negative and no less than -5. 
nInitalALbeta = unifrnd(-5,0,[numInitialsRand,1]);
InitialParamsVec = [repmat(QuantEst',numInitialsRand,1),nInitalALbeta];
RQfval = zeros(numInitialsRand,1);
if doparallel
parfor i = 1:numInitialsRand
    RQfval(i) = ALdist1(InitialParamsVec(i,:),y,Regressor,theta,model,empiricalQuantile);
end
else
for i = 1:numInitialsRand
    RQfval(i) = ALdist1(InitialParamsVec(i,:),y,Regressor,theta,model,empiricalQuantile);
end
end
Results = [RQfval,InitialParamsVec];
SortedResults = sortrows(Results,1);
beta = SortedResults(1:numInitials,2:size(SortedResults,2));
end

%------------------------------------------------------------------------
% function to retrun the -loglikehood of AL dist and condQ and ES
function [llh,VaR,es] = ALdist1(Params,y,X,THETA,MODEL,empiricalQuantile)
%*****************************************************************************************
if ~isempty(X)
if MODEL == 1
    BETA = Params(1:4); 
    phi = Params(5); 
else
    BETA = Params(1:5); 
    phi = Params(6); 
end
else
if MODEL == 1
    BETA = Params(1:3); 
    phi = Params(4); 
else
    BETA = Params(1:4); 
    phi = Params(5); 
end
end
%
%**********************************************
% Compute the VaR

%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1
  VaR = SAVloop(BETA, y, X, empiricalQuantile); 
%
%********************************************************************************************
% Model 2: Asymmetric Slope.
%
else 
   VaR = ASloop(BETA, y, X, empiricalQuantile);
end
%Because the original code of Manganelli is positive VaR
%%
% Compute Asymmetric Laplace loglikelihood
es = (1 + exp(phi)).*VaR;
hit = THETA - (y<=VaR);
ALdistLog = log(((THETA-1)./(es)).*exp(((y-VaR).*hit)./(THETA.*es)));
ALdistLog(~isreal(ALdistLog),:) = -1e100;
llh = -1*sum(ALdistLog);
end

%----------------------------------------------------------------------------------------
% Local function to get simulated ySim - To be used to calculate the
% standard errors, similar to to the Taylor(2017) paper;
function [ySim,CondQsim,CondESsim] = GetSim(BetaHat,Model,FirstY,XSim,FirstCondQ,FirstCondES,ResidSim)
nobs = length(ResidSim); 
CondQsim = zeros(nobs,1);
CondQsim(1) = FirstCondQ;
CondESsim = zeros(nobs,1);
CondESsim(1) = FirstCondES;
ySim = zeros(nobs,1);
ySim(1) = FirstY;
NegResidMean = mean(ResidSim(ResidSim < 0));
if ~isempty(XSim)
if Model == 1
for t = 2:nobs
    CondQsim(t) = BetaHat(1) + BetaHat(2) * CondQsim(t-1) + BetaHat(3) * abs(ySim(t-1)) + BetaHat(4) * XSim(t-1); 
    CondESsim(t) = (1+exp(BetaHat(5)))*CondQsim(t); 
    ySimDay = CondQsim(t) + abs(CondQsim(t))*ResidSim(t);
    if ySimDay <= CondQsim(t)
        ySimDay = CondQsim(t) + (ResidSim(t)/NegResidMean) * (CondESsim(t) - CondQsim(t));
    end
    ySim(t) = ySimDay;
end
else
  for t = 2:nobs
    CondQsim(t) = BetaHat(1) + BetaHat(2) * CondQsim(t-1) + BetaHat(3) * abs(ySim(t-1)) * (ySim(t-1)>0) - BetaHat(4) * abs(ySim(t-1)) * (ySim(t-1)<=0) + BetaHat(5) * XSim(t-1);
    CondESsim(t) = (1+exp(BetaHat(6)))*CondQsim(t); 
    ySimDay = CondQsim(t) + abs(CondQsim(t))*ResidSim(t);
    if ySimDay <= CondQsim(t)
        ySimDay = CondQsim(t) + (ResidSim(t)/NegResidMean) * (CondESsim(t) - CondQsim(t));
    end
    ySim(t) = ySimDay;
   end  
end
else
  if Model == 1
    for t = 2:nobs
    CondQsim(t) = BetaHat(1) + BetaHat(2) * CondQsim(t-1) + BetaHat(3) * abs(ySim(t-1));
    CondESsim(t) = (1+exp(BetaHat(4)))*CondQsim(t); 
    ySimDay = CondQsim(t) + abs(CondQsim(t))*ResidSim(t);
    if ySimDay <= CondQsim(t)
        ySimDay = CondQsim(t) + (ResidSim(t)/NegResidMean) * (CondESsim(t) - CondQsim(t));
    end
    ySim(t) = ySimDay;
    end
  else
    for t = 2:nobs
    CondQsim(t) = BetaHat(1) + BetaHat(2) * CondQsim(t-1) + BetaHat(3) * abs(ySim(t-1)) * (ySim(t-1)>0) - BetaHat(4) * abs(ySim(t-1)) * (ySim(t-1)<=0);
    CondESsim(t) = (1+exp(BetaHat(5)))*CondQsim(t); 
    ySimDay = CondQsim(t) + abs(CondQsim(t))*ResidSim(t);
    if ySimDay <= CondQsim(t)
        ySimDay = CondQsim(t) + (ResidSim(t)/NegResidMean) * (CondESsim(t) - CondQsim(t));
    end
    ySim(t) = ySimDay;
    end  
  end
end
end

%----------------------------------------------------------------------
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