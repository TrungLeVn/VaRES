function [estPars, ht, stdresid, output] = gjr_VaR(y,varargin)

% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes to estimate Gjr-Garch model and make the conditional quantile
% based on simulation
% Procedures: 
% 1. Estimate the gjrGarch model to get the estimated parameters.
% 2. Simulate B-path of returns to compute the conditional VaR and ES. The
%    simulation will depend on the ending date, return horizon, number of 
%    simulations, quantile of interest.
%    2a: Out-of-Sample: Just use the ending of the estimation sample to
%    start the simulation.
%    2b: In-Sample: Need to specify the pseudo ending date. Should depend
%    on the return horizon since we only consider non-overlapping in our
%    application. 
% NOTE: Since we only need to employ simulation once we want to estimate
% forecast for VaR and ES. If not, just return estPars, ht, stdresid. These
% estimates are important in the copula paper. 

% ****************************************************************************************************************************************
quantileDefault = [0.01,0.025,0.05];
periodDefault = 1; 
callerName = 'gjr_VaR';
gjrSpecDefault = [1,1,1];
meanSpecDefault = [0,0]; 
numSimsDefault = 1000;
parseObj = inputParser;
addParameter(parseObj,'Quantile',quantileDefault,@(x)validateattributes(x,{'numeric'},{'nonempty'},callerName));
addParameter(parseObj,'Period',periodDefault,@(x)validateattributes(x,{'numeric'},{'scalar','integer','positive'},callerName));
addParameter(parseObj,'Dates',[],@(x)validateattributes(x,{'numeric','cell'},{},callerName));
addParameter(parseObj,'Display',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'gjrSpec',gjrSpecDefault,@(x)validateattributes(x,{'numeric'},{'column'},{'scalar'},callerName));
addParameter(parseObj,'armaSpec',meanSpecDefault,@(x)validateattributes(x,{'numeric'},{'column'},{'scalar'},callerName));
addParameter(parseObj,'tarchType',2,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'Dist','SKEWT',@(x)validateattributes(x,{'char'},{},callerName));
addParameter(parseObj,'numSims',numSimsDefault,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'DoVar',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'InOrOut',2,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'parallel',false,@(x)validateattributes(x,{'numeric','logical'},{'binary','nonempty'},callerName));
addParameter(parseObj,'cores',4,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'forecastLength',0,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'VaRmethod',1,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'Threshold',0.075,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));
addParameter(parseObj,'RestartVaR',1,@(x)validateattributes(x,{'numeric'},{'scalar'},callerName));

parse(parseObj,varargin{:});
quantileAll         = parseObj.Results.Quantile;
period              = parseObj.Results.Period;
yDates              = parseObj.Results.Dates;
display             = parseObj.Results.Display;
gjrSpec             = parseObj.Results.gjrSpec;
dist                = parseObj.Results.Dist;
dovar               = parseObj.Results.DoVar;
numSims             = parseObj.Results.numSims;
doparallel          = parseObj.Results.parallel;
cores               = parseObj.Results.cores;
inorout             = parseObj.Results.InOrOut; % 1: In-Sample pseudo-simulation / 2: Out-sample simulation.
tarchType           = parseObj.Results.tarchType;
forecastLength      = parseObj.Results.forecastLength;
armaSpec            = parseObj.Results.armaSpec;
varmethod           = parseObj.Results.VaRmethod;
threshold           = parseObj.Results.Threshold;
restartVaR          = parseObj.Results.RestartVaR;
%%
%================================
% INPUT CHECKING
%================================
% Estimate VaR and ES or not? 

if dovar
    if ~ismember(inorout,[1,2])
        error('The simulation type choice needs to be either 1 or 2.')
    end
end

% If doing out-of-sample forecast, how many observations will be hold up?
% Forecast length need to be at least equal to the return horizon

if inorout == 2
    if forecastLength <= 0
       error('Forecast length needs to be provided as a positive value.')
    end
    if forecastLength < period
        error('Forecast length needs to be at least equal to the return horizon.')
    end
end

% Initiate the parallel
if doparallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        parpool('local',cores);
    end
end

%%
%=============================================================================
% DATA ORGANIZE
%=============================================================================
% Replace missing value by the mean of the in-sample estimation
y = y(:);
y(isnan(y)) = nanmean(y);
Nfull = length(y); 

% Load dates
if isempty(yDates)
    yDates = 1:Nfull;
else
    if iscell(yDates)
    yDates = datenum(yDates);    
    end
    if numel(yDates) ~= Nfull
    error('Length of Dates must equal the number of observations.')
    end
end

% Organize data according to InOrOut model specification

Nin = Nfull - forecastLength;
yFull = y; 
yDatesFull = yDates;
y = yFull(1:Nin);
yDates = yDatesFull(1:Nin);

%%
% ========================================================================================
% Mean equation filtering
% ========================================================================================
ar = armaSpec(1); ma = armaSpec(2); 
[meanPars,~,resid,~,~,meanOutput] = armaxfilter(y,1,1:ar,1:ma);
MeanConst = meanPars(1); ARparams = meanPars(2:ar+1); MAparams = meanPars(ar+2:end);
%%
% ========================================================================================
% Estimate the gjrGarch model on the residals - Use the MLE toolbox of
% Sherpard - Oxford
% ========================================================================================

% Get the GARCH specification
p = gjrSpec(1); o = gjrSpec(2); q = gjrSpec(3); 
varOrder = max(gjrSpec); %Maximum order in the GARCH specification. 
meanOrder = max(armaSpec);
m = max(varOrder,meanOrder);

if display
fprintf('Estimating GARCH model....\n');
end

[estPars,LLH,ht,VCV,stdpar,pval,~,score,diagnostic] = tarch(resid,p,o,q,dist,tarchType);
stdresid = resid./sqrt(ht);

% If don't require VaR and ES, just return the GARCH estimates
if ~dovar
    estPars = [meanPars;estPars];
    output.meanEst      = meanOutput;
    output.varPars      = estPars;
    output.CondVar      = ht; 
    output.StdResid     = stdresid; 
    output.LLH          = LLH; 
    output.VCV          = VCV;
    output.CoefStdErr   = stdpar; 
    output.CoefPval     = pval; 
    output.score        = score; 
    output.diagnostic   = diagnostic;
    output.VaR          = [];
    output.ES           = [];
    output.Dates        = [];
    return
end

%%
% ========================================================================================
% Estimating VaR and ES based on simulation if required
% ========================================================================================

if display
fprintf('Making simulation....\n');
end

if inorout == 1
     % If do the in-sample simulation to get the in-sample VaR and ES
   % Specify the date to sim, depends on the return horizon
    N = length(y);
    startdate = max(period,m)+1; % drop the first several initial to start the
   % simulation, depending also on the order in the GARCH specification.
    datetosim = startdate:period:N;
    datetosimNums = length(datetosim);
    VaR = zeros(datetosimNums,length(quantileAll));
    ES = zeros(datetosimNums,length(quantileAll));
    for qloop = 1:length(quantileAll)
    quantile = quantileAll(qloop);
    AllVaR = zeros(datetosimNums,restartVaR);
    AllES = zeros(datetosimNums,restartVaR);
   if doparallel
   parfor ii = 1:restartVaR
        tempVaR = zeros(datetosimNums,1);
        tempES = zeros(datetosimNums,1);
        for t = 1:datetosimNums
        startValsVar = [resid(datetosim(t)-1:-1:datetosim(t)-varOrder),ht(datetosim(t)-1:-1:datetosim(t)-varOrder)];
        startValsMean = [y(datetosim(t)-1:-1:datetosim(t) - meanOrder),resid(datetosim(t)-1:-1:datetosim(t)-meanOrder)];
        ARlags = startValsMean(1:ar,1); MAlags = startValsMean(1:ma,2);
        residSim = tarch_simulate2(period,numSims,estPars,p,o,q,startValsVar,dist);
        ySim = armaxfilter_simulate2(residSim,MeanConst,ar,ARparams,ARlags,ma,MAparams,MAlags);    
        ySimPeriod = sum(ySim,1);% Construct the simulated h-horizon return.
        dataSimSorted = sort(ySimPeriod);    
        if varmethod == 1 % Compute VaR based solely on simulation
        tempVaR(t,1) = dataSimSorted(quantile*numSims);
        exceedances = dataSimSorted(dataSimSorted <= tempVaR(t,1));
        tempES(t,1) = mean(exceedances);
        else %Compute VaR and ES based on EVT and Generalized Pareto Distribution
        thresholdLevel = dataSimSorted(threshold*numSims); 
        StdSim = ySimPeriod./thresholdLevel - 1;  
        StdExtreme = StdSim(StdSim > 0);
        EVTparams = gpfit(StdExtreme); 
        gamma = EVTparams(1); beta = EVTparams(2);
        Nexceed = length(StdExtreme);
        quantileStdExceed = ((quantile*(numSims/Nexceed)).^(-gamma)-1)*(beta/gamma); % The EVT quantile of standardized simulation
        tempVaR(t,1) = thresholdLevel * (1 + quantileStdExceed); % The EVT quantile of simulated data - convert back from standardized value
        tempES(t,1) = tempVaR(t,1) * (1/(1-gamma) + (beta - gamma*thresholdLevel)/(1-gamma)); % Look at equation 21 - Mangenelli and Engle 2004
        end
        end
        AllVaR(:,ii) = tempVaR;
        AllES(:,ii) = tempES;
   end
   else
    for ii = 1:restartVaR
      tempVaR = zeros(datetosimNums,1);
      tempES = zeros(datetosimNums,1);
      for t = 1:datetosimNums
        startValsVar = [resid(datetosim(t)-1:-1:datetosim(t)-varOrder),ht(datetosim(t)-1:-1:datetosim(t)-varOrder)];
        startValsMean = [y(datetosim(t)-1:-1:datetosim(t) - meanOrder),resid(datetosim(t)-1:-1:datetosim(t)-meanOrder)];
        ARlags = startValsMean(1:ar,1); MAlags = startValsMean(1:ma,2);
        residSim = tarch_simulate2(period,numSims,estPars,p,o,q,startValsVar,dist);
        ySim = armaxfilter_simulate2(residSim,MeanConst,ar,ARparams,ARlags,ma,MAparams,MAlags);    
        ySimPeriod = sum(ySim,1);% Construct the simulated h-horizon return.
        dataSimSorted = sort(ySimPeriod);    
        if varmethod == 1  % Compute VaR based solely on simulation
        tempVaR(t,1) = dataSimSorted(quantile*numSims);
        exceedances = dataSimSorted(dataSimSorted <= tempVaR(t,1));
        tempES(t,1) = mean(exceedances);
        else %Compute VaR and ES based on EVT and Generalized Pareto Distribution
        thresholdLevel = dataSimSorted(threshold*numSims); 
        StdSim = ySimPeriod./thresholdLevel - 1; 
        StdExtreme = StdSim(StdSim > 0);
        EVTparams = gpfit(StdExtreme); 
        gamma = EVTparams(1); beta = EVTparams(2);
        Nexceed = length(StdExtreme);
        quantileStdExceed = ((quantile*(numSims/Nexceed)).^(-gamma)-1)*(beta/gamma); % The EVT quantile of standardized simulation
        tempVaR(t,1) = thresholdLevel * (1 + quantileStdExceed); % The EVT quantile of simulated data - convert back from standardized value
        tempES(t,1) = tempVaR(t,1) * (1/(1-gamma) + (beta - gamma*thresholdLevel)/(1-gamma)); 
        end
      end
      AllVaR(:,ii) = tempVaR;
      AllES(:,ii) = tempES;
   end
   end
    VaR(:,qloop) = mean(AllVaR,2); 
    ES(:,qloop) = mean(AllES,2);
    end
    yDates = yDates(datetosim);
else
    % The out-of-sample forecast will be based on pseudo ending points.
    % For example, if the parameters are estimated using first Nin
    % observations and the hold back observations > forecast horizon, the
    % hold back observatioins will first filtered using estimated arma and
    % var parameters. There, the out-of-sample forecasts will depends on
    % real data at the pseudo-ending points.
    VaR = zeros(forecastLength/period,length(quantileAll));
    ES = zeros(forecastLength/period,length(quantileAll));
    for qloop = 1:length(quantileAll)
    quantile = quantileAll(qloop);
    [~,~,residFull] = armaxfilter(yFull,1,1:ar,1:ma,[],[],[],[],[],meanPars); % Get the resid based on armafilter
    [~,~,htFull] = tarch(residFull,p,o,q,dist,tarchType,[],[],estPars,Nin); % Get the ht based on arma + tarch filter
    residForSample = residFull(Nin-m+1:end); 
    htForSample = htFull(Nin-m+1:end);
    yForSample = yFull(Nin-m+1:end);   
    yDatesForSample = yDatesFull(Nin-m+1:end);
    startdate = m+1;
    datetosim = startdate:period:length(residForSample); % Pseudo-ending points
    datetosimNums = length(datetosim);
    % Because the VaR and Estimate seems to be affected by the random
    % numbers, we estimate the VaR and ES for 100 times and then estimate
    % VaR and ES as the mean of all simulated VaR and ES approximation
    AllVaR = zeros(datetosimNums,restartVaR);
    AllES = zeros(datetosimNums,restartVaR);
    if doparallel
    parfor ii = 1:restartVaR
        tempVaR = zeros(datetosimNums,1);
        tempES = zeros(datetosimNums,1);
        for t = 1:datetosimNums
        startValsVar = [residForSample(datetosim(t)-1:-1:datetosim(t)-varOrder),htForSample(datetosim(t)-1:-1:datetosim(t)-varOrder)];
        startValsMean = [yForSample(datetosim(t)-1:-1:datetosim(t) - meanOrder),residForSample(datetosim(t)-1:-1:datetosim(t)-meanOrder)];
        ARlags = startValsMean(1:ar,1); MAlags = startValsMean(1:ma,2);
        residSim = tarch_simulate2(period,numSims,estPars,p,o,q,startValsVar,dist);
        ySim = armaxfilter_simulate2(residSim,MeanConst,ar,ARparams,ARlags,ma,MAparams,MAlags);
        ySimPeriod = sum(ySim,1);
        dataSimSorted = sort(ySimPeriod);
        if varmethod == 1  % Compute VaR based solely on simulation
        tempVaR(t,1) = dataSimSorted(quantile*numSims);
        exceedances = dataSimSorted(dataSimSorted <= tempVaR(t,1));
        tempES(t,1) = mean(exceedances);
        else %Compute VaR and ES based on EVT and Generalized Pareto Distribution
        thresholdLevel = dataSimSorted(threshold*numSims);
        StdSim = ySimPeriod./thresholdLevel - 1; 
        StdExtreme = StdSim(StdSim > 0);
        EVTparams = gpfit(StdExtreme); 
        gamma = EVTparams(1); beta = EVTparams(2);
        Nexceed = length(StdExtreme);
        quantileStdExceed = ((quantile*(numSims/Nexceed)).^(-gamma)-1)*(beta/gamma); % The EVT quantile of standardized simulation
        tempVaR(t,1) = thresholdLevel * (1 + quantileStdExceed); % The EVT quantile of simulated data - convert back from standardized value
        tempES(t,1) = tempVaR(t,1) * (1/(1-gamma) + (beta - gamma*thresholdLevel)/(1-gamma)); 
        end
        end
        AllVaR(:,ii) = tempVaR;
        AllES(:,ii) = tempES;
    end
    else
    for ii = 1:restartVaR
        tempVaR = zeros(datetosimNums,1);
        tempES = zeros(datetosimNums,1);
        for t = 1:datetosimNums
        startValsVar = [residForSample(datetosim(t)-1:-1:datetosim(t)-varOrder),htForSample(datetosim(t)-1:-1:datetosim(t)-varOrder)];
        startValsMean = [yForSample(datetosim(t)-1:-1:datetosim(t) - meanOrder),residForSample(datetosim(t)-1:-1:datetosim(t)-meanOrder)];
        ARlags = startValsMean(1:ar,1); MAlags = startValsMean(1:ma,2);
        residSim = tarch_simulate2(period,numSims,estPars,p,o,q,startValsVar,dist);
        ySim = armaxfilter_simulate2(residSim,MeanConst,ar,ARparams,ARlags,ma,MAparams,MAlags);
        ySimPeriod = sum(ySim,1);
        dataSimSorted = sort(ySimPeriod);
        if varmethod == 1  % Compute VaR based solely on simulation
        tempVaR(t,1) = dataSimSorted(quantile*numSims);
        exceedances = dataSimSorted(dataSimSorted <= tempVaR(t,1));
        tempES(t,1) = mean(exceedances);
        else %Compute VaR and ES based on EVT and Generalized Pareto Distribution
        thresholdLevel = dataSimSorted(threshold*numSims);
        StdSim = ySimPeriod./thresholdLevel - 1; 
        StdExtreme = StdSim(StdSim > 0);
        EVTparams = gpfit(StdExtreme);
        gamma = EVTparams(1); beta = EVTparams(2);
        Nexceed = length(StdExtreme);
        quantileStdExceed = ((quantile*(numSims/Nexceed)).^(-gamma)-1)*(beta/gamma); % The EVT quantile of standardized simulation
        tempVaR(t,1) = thresholdLevel * (1 + quantileStdExceed); % The EVT quantile of simulated data - convert back from standardized value
        tempES(t,1) = tempVaR(t,1) * (1/(1-gamma) + (beta - gamma*thresholdLevel)/(1-gamma)); 
        end
        end
        AllVaR(:,ii) = tempVaR;
        AllES(:,ii) = tempES;
    end   
    end
    VaR(:,qloop) = mean(AllVaR,2); 
    ES(:,qloop) = mean(AllES,2);
    end
    yDates = yDatesForSample(datetosim);
end
%%
%**************************** Store the outputs in the vector 'output' ******************************************
estPars = [meanPars;estPars];
output.meanEst      = meanOutput;
output.estPars      = estPars;
output.CondVar      = ht; 
output.StdResid     = stdresid; 
output.LLH          = LLH; 
output.VCV          = VCV;
output.CoefStdErr   = stdpar; 
output.CoefPval     = pval; 
output.score        = score; 
output.diagnostic   = diagnostic;
output.VaR          = VaR;
output.ES           = ES;
output.Dates        = yDates;
end