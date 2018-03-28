    function [estParams,CondQ,CondES,output] = VarEs_Ols(y,varargin)
%VARES Summary of this function goes here
%   Detailed explanation goes here
quantileDefault = 0.05;
periodDefault = 1;
nlagDefault = 100;
callerName = 'VarEs_Ols2';
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'scalar','>',0,'<',1},callerName));
addParameter(parseObj,'X',[],@(x)validateattributes(x,{'numeric'},{'2d'},callerName));
addParameter(parseObj,'Model',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'NumLags',nlagDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer'},callerName));
addParameter(parseObj,'Ovlap',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'SubModel',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'DoParallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Cores',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'Params',[],@(x)validateattributes(x,{'numeric'},{'column'},callerName));
addParameter(parseObj,'xDates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'GetSe',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'Constrained',true,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'mu0',[],@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'startPars',[]);

%%
parse(parseObj,varargin{:});
theta = parseObj.Results.Quantile;
period = parseObj.Results.Period;
nlag = parseObj.Results.NumLags;
ovlap = parseObj.Results.Ovlap;
Regressor = parseObj.Results.X;
model = parseObj.Results.Model;
submodel = parseObj.Results.SubModel; 
doparallel = parseObj.Results.DoParallel;
cores = parseObj.Results.Cores;
yDates = parseObj.Results.Dates;
xDates = parseObj.Results.xDates;
BetaHat = parseObj.Results.Params;
getse = parseObj.Results.GetSe;
constrained = parseObj.Results.Constrained;
startPars = parseObj.Results.startPars;

mu0 =  parseObj.Results.mu0;
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
    startPars = startPars(1:end-1);
end
%%
switch model
case 1  % Model = 1: using CAViaR as the univariate quantile estimate

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
   [~,CondQ] = CAViaR_X(y,'Params',BetaHat(1:(end-1)),'X',Regressor,'xDates',xDates,'Quantile',theta,...
       'Model',submodel,'Display',false,'GetSe',false,'Dates',yDates,'DoFull',true,...
       'NumLags',nlag,'Constrained',constrained,'Period',period);
    else
    [~,CondQ] = CAViaR_Uni(y,'Params',BetaHat(1:(end-1)),'Quantile',theta,...
       'Model',submodel,'Display',false,'GetSe',false,'Dates',yDates,'DoFull',true,...
       'NumLags',nlag,'Constrained',constrained,'Period',period);    
    end
    CondES = BetaHat(end) * CondQ;
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
    end
    return
end
 
% Optimization procedures
    fprintf('Estimating univariate CAViaR estimate to get initial parameters\n');
    if ~isempty(Regressor)
    [QuantEst,CondQ,VaRoutput] = CAViaR_X(y,'Dates',yDates,'X',Regressor','xDates',xDates,'Quantile',theta,...
        'Model',submodel,'Display',false,'GetSe',getse,'DoParallel',doparallel,'Cores',cores,...
        'DoFull',true,'Constrained',constrained,'NumLags',nlag,'Period',period,'startPars',startPars);
    else
    [QuantEst,CondQ,VaRoutput] = CAViaR_Uni(y,'Dates',yDates,'Quantile',theta,...
        'Model',submodel,'Display',false,'GetSe',getse,'DoParallel',doparallel,'Cores',cores,...
        'DoFull',true,'Constrained',constrained,'NumLags',nlag,'Period',period,'startPars',startPars);   
    end
% OLS estimations for the expected shortfall equation
fprintf('Estimating expected shortfall equation...\n');
Exceed = y(y <= CondQ); 
VaRexceed = CondQ(y<=CondQ);
%[ESparams,~,~,ESstd,~,R2] = ols(Exceed,VaRexceed,1);
ESparams = VaRexceed\Exceed;
CondES = ESparams*CondQ;
case 2 % Model = 2: using MIDAS as the univariate quantile estimate
    
    % Load the conditioning variable (predictor)
if isempty(Regressor)
   Regressor = abs(y);
   if isempty(xDates)
       xDates = yDates;
   end
end
    
    % If the model parameters are provided, just estimate the CondQ and CondES
if ~isempty(BetaHat)
    if submodel == 1
    [~,CondQ,VaRoutput] = MidasQuantile(y,'Dates',yDates','X',Regressor,'Quantile',theta,'xDates',xDates,'Params',BetaHat(1:(end-1)),'Display',false,'GetSe',false,...
        'Ovlap',ovlap,'Period',period,'NumLags',nlag,'Constrained',constrained);
    else
    [~,CondQ,VaRoutput] = MidasQuantileAS(y,'Dates',yDates','X',Regressor,'Quantile',theta,'xDates',xDates,'Params',BetaHat(1:(end-1)),'Display',false,'GetSe',false,...
        'Ovlap',ovlap,'Period',period,'NumLags',nlag,'Constrained',constrained);
    end
    CondES = BetaHat(end) * CondQ;
    y = VaRoutput.yLowFreq;
    yDates = VaRoutput.yDates;
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
    end
    return
end

% First get the parameter estimates of the quantile regression
  
fprintf('Estimating univariate MIDAS estimate to get initial parameters\n');
    if submodel == 1
    [QuantEst,CondQ,VaRoutput] = MidasQuantile(y,'Dates',yDates','X',Regressor,'Quantile',theta,'Display',false,'GetSe',getse,...
            'Ovlap',ovlap,'xDates',xDates,'Period',period,'NumLags',nlag,'DoParallel',doparallel,'Cores',cores,'Constrained',constrained,'startPars',startPars);
    else
    [QuantEst,CondQ,VaRoutput] = MidasQuantileAS(y,'Dates',yDates','X',Regressor,'Quantile',theta,'Display',false,'GetSe',getse,...
            'Ovlap',ovlap,'xDates',xDates,'Period',period,'NumLags',nlag,'DoParallel',doparallel,'Cores',cores,'Constrained',constrained,'startPars',startPars);
    end
% OLS estimations for the expected shortfall equation
fprintf('Estimating expected shortfall equation...\n');

% Mix data to get the OLS regression
y = VaRoutput.yLowFreq;
yDates = VaRoutput.yDates;

% Compute CondES using OLS regression coefficient
Exceed = y(y <= CondQ); 
VaRexceed = CondQ(y<=CondQ);
%[ESparams,~,~,ESstd,~,R2] = ols(Exceed,VaRexceed,1);
ESparams = VaRexceed\Exceed;
CondES = ESparams*CondQ;
end

%% Get output report
% Get the full parameters: CAViaR and ES Ols regression
estParams = [QuantEst;ESparams];
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
%output.VaRoutput = VaRoutput;
output.quantile = theta;
output.VaR = CondQ;
output.ES = CondES;
%output.ESstd = ESstd;
%output.ESR2 = R2;
output.Dates = yDates;
output.y = y;
output.Regressor = Regressor;
output.exitFlag = VaRoutput.exitFlag;
end
end

%%
% Local Function
%-------------------------------------------------------
% Local function for OLS estimation
function [estimates, errors, estvar, stderror,Tvalues,R2,R3,robustV,robustSE,robustT]=ols(y,x,noconst)

% usage: [estimates, error, estvar, stderror,Tvalues,R2,R2adj]=ols(y,x,noconst)
%the ols(y,x, noconst) function takes a vector and
%a matrix (vector)as arguments.
%The rows of the arguments must be identical
%The y argument is the vector of of dependent variable
%The x argument is the matrix of independent variables
%The x matrix must NOT include a column of ones.
%The regression is run with a constant by default.
%IF the noconst parameter is 1, the regression is run WITHOUT a constant
%IF additional results such as errrors, estimated variance, or R2 are
%requested, they must be specified in the output matrix.

%copyright Rossen Valkanov, 02/22/96
%Last modified Arthur Sinko 9/12/2009
[r,t]=size(x); [v, w]=size(y);
if nargin<3 , noconst=0; end;
if noconst==0, xr=[ones(r,1) x]; 	  %run with a constant
else xr=x;end				              % NO constant
[r,t]=size(xr);
indic=0;
if r<=t
    indic=NaN;
end
if r~=v
    error('Incompatible matices');
end
invXpX=inv(xr'*xr);
P=invXpX*xr'+indic;	%projection matrix
estimates=P*y;       %ols estimates


if nargout>1 		%execute only if more than estimates are required
    degfr=r-t;
    yhat=xr*estimates;
    errors=y-yhat;
    RSS=errors'*errors;
    s2=RSS./degfr;
    estvar=s2*invXpX;
    stderror=sqrt(diag(estvar));
    TSS=(y-mean(y))'*(y-mean(y));			  %Total SS
    R2=1-RSS/TSS;                         %R2
    R3=1-((r-1)/(r-t))*(1-R2);            %adj R2--note: it might be negative or greater than zero, use only when many variables
    Tvalues=estimates./stderror; %tvalues for null coeff=0.
% end;
% 
    if nargout>7
    robustV=r/degfr*P*HAC_kernel(errors,'NW')*P';
    robustSE=sqrt(diag(robustV));
    robustT=estimates./robustSE; 
    end
end
end

% Local function 
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