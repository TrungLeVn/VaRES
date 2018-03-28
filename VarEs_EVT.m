function [estParams,CondQ,CondES,output] = VarEs_EVT(y,varargin)
%VARES Summary of this function goes here
%   Detailed explanation goes here
quantileDefault = 0.05;
periodDefault = 1;
nlagDefault = 100;
callerName = 'VarEs_EVT';
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'scalar','>',0,'<',1},callerName));
addParameter(parseObj,'X',[],@(x)validateattributes(x,{'numeric'},{'2d'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'NumLags',nlagDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer'},callerName));
addParameter(parseObj,'xDates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'DoParallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Cores',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Ovlap',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Model',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'SubModel',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'Params',[],@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'GetSe',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Constrained',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Threshold',0.075,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'mu0',[],@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'startPars',[]);

parse(parseObj,varargin{:});
theta = parseObj.Results.Quantile;
period = parseObj.Results.Period;
nlag = parseObj.Results.NumLags;
Regressor = parseObj.Results.X;
model = parseObj.Results.Model;
submodel = parseObj.Results.SubModel; 
doparallel = parseObj.Results.DoParallel;
cores = parseObj.Results.Cores;
ovlap = parseObj.Results.Ovlap;
yDates = parseObj.Results.Dates;
xDates = parseObj.Results.xDates;
BetaHat = parseObj.Results.Params;
getse = parseObj.Results.GetSe;
constrained = parseObj.Results.Constrained;
threshold = parseObj.Results.Threshold;
mu0 = parseObj.Results.mu0;
startPars = parseObj.Results.startPars;

if ~ismember(model,[1,2])
    disp(' ')
    disp('*******************************************************************')
    disp('ERROR! You need to select one of the following models:')
    disp('Model=1: CAViaR')
    disp('Model=2: MIDAS')
end

% Replace missing values by the sample average
if isempty(mu0)
    mu0 = 0;
end

y = y(:);
y(isnan(y)) = nanmean(y);
nobs = length(y);
y = y - mu0;

% Load the conditioning variable (predictor)
if isempty(yDates)
    yDates = 1:nobs;
else
if numel(yDates) ~= nobs
    error('Length of Dates must equal the number of observations.')
end
end

if doparallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local',cores);
    end
end

%%
% Reshape the startPars to include only univariate quantile estimates
if ~isempty(startPars)
    startPars = startPars(1:end-2);
end
%%
switch model
    case 1
        
% Mix the data to fit the return horizon and nlag similar to the MIDAS model
ovlap = false; 
MixedData = MixedFreqQuantile(y,yDates,y,yDates,nlag,period,ovlap);
y = MixedData.EstY;
yDates = MixedData.EstYdate;

% Load the conditioning variable (predictor)
if ~isempty(Regressor)
   MixedVar = MixedFreqQuantile(Regressor,xDates,Regressor,xDates,nlag,period,ovlap);
   Regressor = sqrt(MixedVar.EstY);
   xDates = MixedData.EstYdate;
end

% If the model parameters are provided, just estimate the CondQ and CondES
if ~isempty(BetaHat)
    if ~isempty(Regressor)
   [~,CondQthreshold] = CAViaR_X(y,'Dates',yDates,'X',Regressor','xDates',xDates,'Model',submodel,'Quantile',threshold,...
       'Display',false,'GetSe',false,'Params',BetaHat(1:(end-2)),'DoFull',true,'Constrained',constrained,...
       'Period',period,'NumLags',nlag);
    else
    [~,CondQthreshold] = CAViaR_Uni(y,'Dates',yDates,'Model',submodel,'Quantile',threshold,...
       'Display',false,'GetSe',false,'Params',BetaHat(1:(end-2)),'DoFull',true,'Constrained',constrained,...
       'Period',period,'NumLags',nlag);
    end
    StdExceed = y./CondQthreshold - 1;
    Extreme = StdExceed(StdExceed >0);
    Nexceed = length(Extreme);
    nobs = length(y);
    gamma = BetaHat(end-1); beta = BetaHat(end);
    StdQuantile = (((theta*(nobs/Nexceed)).^(-gamma)-1)*(beta/gamma)); % The EVT quantile of standardized quantile 
    CondQ = CondQthreshold .* (1 + StdQuantile); % Back to quantile
    ExpectedMean_StdQuant = StdQuantile .* (1/(1-gamma) + beta./((1-gamma).*StdQuantile));
    CondES = CondQthreshold .* (1 + ExpectedMean_StdQuant);
    estParams = BetaHat;
    Hit = theta - (y <= CondQ);
    nobs = length(Hit);
    HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
    CondQ = CondQ + period * mu0;
    CondES = CondES + period * mu0;
    y = y + period * mu0;
    if nargout > 3
    output.estParams = estParams;
    output.Hit = Hit; 
    output.HitPercentage = HitPercentage;
    output.quantile = theta;
    output.VaR = CondQ;
    output.ES = CondES;
    output.Dates = yDates;
    output.y = y;
    output.Regressor = Regressor;
    output.CondQthreshold = CondQthreshold;
    end
    return    
end

% Optimization procedures
    fprintf('Estimating univariate CAViaR estimate to get initial parameters\n');
    if ~isempty(Regressor)
    [ThresholdParams,CondQthreshold,VaRoutput] = CAViaR_X(y,'Dates',yDates,'X',Regressor','xDates',xDates,'Model',submodel,...
        'Quantile',threshold,'Display',false,'GetSe',getse,'DoParallel',doparallel,'Cores',cores,'DoFull',true,...
        'Period',period','NumLags',nlag,'Constrained',constrained,'startPars',startPars);
    else 
    [ThresholdParams,CondQthreshold,VaRoutput] = CAViaR_Uni(y,'Dates',yDates,'Model',submodel,...
        'Quantile',threshold,'Display',false,'GetSe',getse,'DoParallel',doparallel,'Cores',cores,'DoFull',true,...
        'Period',period','NumLags',nlag,'Constrained',constrained,'startPars',startPars);     
    end
% Start the optimization procedures
fprintf('Estimating the Generalized Pareto Distribution...\n');
StdExceed = y./CondQthreshold - 1;
Extreme = StdExceed(StdExceed >0);
EVTparams = gpfit(Extreme); % Fit the Generalize Pareto Distribution
gamma = EVTparams(1); beta = EVTparams(2);
Nexceed = length(Extreme);
nobs = length(y);
StdQuantile = ((theta*(nobs/Nexceed)).^(-gamma)-1)*(beta/gamma);
CondQ = CondQthreshold .* (1 + StdQuantile);
ExpectedMean_StdQuant = StdQuantile .* (1/(1-gamma) + beta./((1-gamma).*StdQuantile));
CondES = CondQthreshold .* (1 + ExpectedMean_StdQuant);

    case 2

if isempty(Regressor)
   Regressor = abs(y);
    if isempty(xDates)
       xDates = yDates;
   end
end

% If the model parameters are provided, just estimate the CondQ and CondES
if ~isempty(BetaHat)
if submodel == 1
   [~,CondQthreshold,VaRoutput] = MidasQuantile(y,'Dates',yDates','X',Regressor,'Quantile',threshold,...
       'xDates',xDates,'Display',false,'GetSe',false,'Params',BetaHat(1:(end-2)),...
            'Ovlap',ovlap,'Period',period,'NumLags',nlag,'Constrained',constrained);
else
    [~,CondQthreshold,VaRoutput] = MidasQuantileAS(y,'Dates',yDates','X',Regressor,'Quantile',threshold,...
        'xDates',xDates,'Display',false,'GetSe',false,'Params',BetaHat(1:(end-2)),...
            'Ovlap',ovlap,'Period',period,'NumLags',nlag,'Constrained',constrained);
end
    y = VaRoutput.yLowFreq;
    yDates = VaRoutput.yDates;
    StdExceed = y./CondQthreshold - 1;
    Extreme = StdExceed(StdExceed >0);
    Nexceed = length(Extreme);
    nobs = length(y);
    gamma = BetaHat(end-1); beta = BetaHat(end);
    StdQuantile = (((theta*(nobs/Nexceed)).^(-gamma)-1)*(beta/gamma));
    CondQ = CondQthreshold .* (1 + StdQuantile);
    ExpectedMean_StdQuant = StdQuantile .* (1/(1-gamma) + beta./((1-gamma).*StdQuantile));
    CondES = CondQthreshold .* (1 + ExpectedMean_StdQuant);   
    estParams = BetaHat;
    Hit = theta - (y <= CondQ);
    nobs = length(Hit);
    HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
    CondQ = CondQ + period * mu0;
    CondES = CondES + period * mu0;
    y = y + period * mu0;
    if nargout > 3
    output.estParams = estParams;
    output.Hit = Hit; 
    output.HitPercentage = HitPercentage;
    output.quantile = theta;
    output.VaR = CondQ;
    output.ES = CondES;
    output.Dates = yDates;
    output.y = y;
    output.Regressor = Regressor;
    output.VaRoutput = VaRoutput;
    end
    return
end
% Optimization procedures
    fprintf('Estimating univariate MIDAS estimate to get initial parameters\n');
    if submodel == 1
    [ThresholdParams,CondQthreshold,VaRoutput] = MidasQuantile(y,'Dates',yDates','X',Regressor,'Quantile',threshold,'Display',false,'GetSe',false,...
            'Ovlap',ovlap,'xDates',xDates,'Period',period,'NumLags',nlag,'DoParallel',doparallel,'Cores',cores,'Constrained',constrained,'startPars',startPars);
    else
    [ThresholdParams,CondQthreshold,VaRoutput] = MidasQuantileAS(y,'Dates',yDates','X',Regressor,'Quantile',threshold,'Display',false,'GetSe',false,...
            'Ovlap',ovlap,'xDates',xDates,'Period',period,'NumLags',nlag,'DoParallel',doparallel,'Cores',cores,'Constrained',constrained,'startPars',startPars);
    end
% Start the optimization procedures
    fprintf('Estimating the Generalized Pareto Distribution...\n');
    y = VaRoutput.yLowFreq;
    yDates = VaRoutput.yDates;
    nobs = length(y);
    StdExceed = y./CondQthreshold - 1;
    Extreme = StdExceed(StdExceed > 0);
    EVTparams = gpfit(Extreme);
    gamma = EVTparams(1); beta = EVTparams(2);
    Nexceed = length(Extreme);
    StdQuantile = (((theta*(nobs/Nexceed)).^(-gamma)-1)*(beta/gamma));
    CondQ = CondQthreshold .* (1 + StdQuantile); % Back to quantile
    ExpectedMean_StdQuant = StdQuantile .* (1/(1-gamma) + beta./((1-gamma).*StdQuantile));
    CondES = CondQthreshold .* (1 + ExpectedMean_StdQuant);
end

%% Get output report
% Get the full parameters: CAViaR and ES Ols regression
Hit = theta - (y <= CondQ);
nobs = length(Hit);
HitPercentage    = mean(Hit(1:nobs) + theta) * 100;
estParams = [ThresholdParams;EVTparams'];
CondQ = CondQ + period * mu0;
CondES = CondES + period * mu0;
y = y + period * mu0;
exitFlag = VaRoutput.exitFlag;
if EVTparams(1) >= 1||any(CondQ > 0)
    exitFlag = 0; 
end
output.estParams= estParams;
output.Hit = Hit; 
output.HitPercentage = HitPercentage;
output.quantile = theta;
output.VaR = CondQ;
output.ES = CondES;
output.Dates = yDates;
output.y = y;
output.Regressor = Regressor;
output.exitFlag = exitFlag;
output.VaRoutput = VaRoutput;
output.CondQthreshold = CondQthreshold;
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