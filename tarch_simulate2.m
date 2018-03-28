function [simulatedata, ht] = tarch_simulate2(t,numSims,parameters,p,o,q,startVals, error_type,tarch_type)
% TARCH(P,O,Q) time series simulation with multiple error distributions
%
% USAGE:
%   [SIMULATEDATA, HT] = tarch_simulate(T, PARAMETERS, P, O, Q, ERROR_TYPE, TARCH_TYPE)
%
% INPUTS:
%   T             - Length of the time series to be simulated  OR
%                   T by 1 vector of user supplied random numbers (i.e. randn(1000,1))
%   PARAMETERS    - a 1+P+O+Q (+1 or 2, depending on error distribution) x 1 parameter vector
%                   [omega alpha(1) ... alpha(p) gamma(1) ... gamma(o) beta(1) ... beta(q) [nu lambda]]'.
%   P             - Positive, scalar integer representing the number of symmetric innovations
%   O             - Non-negative scalar integer representing the number of asymmetric innovations (0
%                     for symmetric processes) 
%   Q             - Non-negative, scalar integer representing the number of lags of conditional
%                     variance (0 for ARCH) 
%   startVals   - 2 by m matrix in which the first column is the start
%                   values for StdResid and the second column is the start
%                   values for h_t. 
%   ERROR_TYPE    - [OPTIONAL] The error distribution used, valid types are:
%                     'NORMAL'    - Gaussian Innovations [DEFAULT]
%                     'STUDENTST' - T distributed errors
%                     'GED'       - Generalized Error Distribution
%                     'SKEWT'     - Skewed T distribution
%   TARCH_TYPE    - [OPTIONAL] The type of variance process, either
%                     1 - Model evolves in absolute values
%                     2 - Model evolves in squares [DEFAULT]
%
% OUTPUTS:
%   SIMULATEDATA  - A time series with ARCH/GARCH/GJR/TARCH variances
%   HT            - A vector of conditional variances used in making the time series
%
% COMMENTS:
% The conditional variance, h(t), of a TARCH(P,O,Q) process is modeled as follows:
%     g(h(t)) = omega
%             + alpha(1)*f(r_{t-1}) + ... + alpha(p)*f(r_{t-p})
%             + gamma(1)*I(t-1)*f(r_{t-1}) +...+ gamma(o)*I(t-o)*f(r_{t-o})
%             + beta(1)*g(h(t-1)) +...+ beta(q)*g(h(t-q))
%
%     where f(x) = abs(x)  if tarch_type=1
%           g(x) = sqrt(x) if tarch_type=1
%           f(x) = x^2     if tarch_type=2
%           g(x) = x       if tarch_type=2
%
% NOTE: This program generates 2000 more than required to minimize any starting bias
%
% See also TARCH, EGARCH_SIMULATE, APARCH_S

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 3    Date: 9/1/2005

% NOTE: Modification on 07/02/2018
% In our application, instead of using the unconditional volatily and zero for std
% resid, we use the current resid and conditional volatility to start to simulation. 
% Morever, since we use real data, we do
% not need to generate 2000 more observation in the burn in sample. In
% other words, all the simulated path start by the same volatility and
% resid, but will be different depending on the random sample.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==7
    error_type='NORMAL';
    tarch_type=2;
elseif nargin==8
    tarch_type=2; 
elseif nargin==9
    %nothing
else
    error('6 to 9 inputs only.');
end

if isempty(error_type)
    error_type='NORMAL';
end

if strcmp(error_type,'NORMAL')
    extrap=0;
elseif strcmp(error_type,'STUDENTST')
    extrap=1;
elseif strcmp(error_type,'GED')
    extrap=1;
elseif strcmp(error_type,'SKEWT')
    extrap=2;
else
    error('Unknown ERROR_TYPE')
end

if ~(tarch_type==1 || tarch_type==2)
    error('TARCH_TYPE must be either 1 or 2')
end

if size(parameters,2)>size(parameters,1)
    parameters = parameters';
end

if size(parameters,1)~=(1+p+o+q+extrap) || size(parameters,2)>1
    error('PARAMETERS must be a column vector with the correct number of parameters.');
end

if (sum(parameters(2:p+1)) + 0.5*sum(parameters(p+2:p+o+1)) + sum(parameters(p+o+2:p+o+q+1))) >=1
    warning('UCSD_GARCH:nonstationary','PARAMETERS are in the non-stationary space, be sure to check that H is not inf.');
end

if length(p)>1 || length(o)>1 || length(q)>1 || any(q<0) || any(p<1) || any(o<0)
    error('P, O and Q must be scalars with P positive and O and Q non-negative')
end

if size(t,2)~=1
    error('T must be either a positive scalar or a vector of random numbers.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Separate the parameters
omega=parameters(1);
alpha=parameters(2:p+1);
gamma=parameters(p+2:p+o+1);
beta=parameters(p+o+2:p+o+q+1);

% Initiate start value for standardized residuals and conditional volatility. 

% TRUNG: Need to check the startvals to be consistent with the lag order
startResid = startVals(:,1); 
startVol   = startVals(:,2);

%Initialize the random numbers
if isscalar(t)
    %t=t+2000; %Since we use the current volatility and residuals, we do not
    %need to make 2000 aditional simulation as a burn-in sample.
    % Instead of making only one simulation, which seems to be highly effected
    % by the random number, we make 1000 paths, then take the mean at each time
    % t to get the final simulation
    if strcmp(error_type,'NORMAL')
        RandomNums=randn(t,numSims);
    elseif strcmp(error_type,'STUDENTST')
        nu=parameters(2+o+p+q);
        RandomNums=stdtrnd(nu,t,numSims);
    elseif strcmp(error_type,'GED')
        nu=parameters(2+o+p+q);
        RandomNums=gedrnd(nu,t,numSims);
    elseif strcmp(error_type,'SKEWT')
        nu=parameters(2+o+p+q);
        lambda=parameters(3+o+p+q);
        RandomNums=skewtrnd(nu,lambda,t,numSims); %Generate standardized Skew-t - to be used as z_t in mean equation
    else
        error('Unknown error type')
    end
else
    RandomNums=t;
    t=size(RandomNums,numSims);
    seeds=ceil(rand(2000,numSims).*t);
    RandomNums=[RandomNums(seeds);RandomNums];
    t=size(RandomNums,1);
end

m  =  max([p,o,q]);
%Back casts, zeros are fine since we are throwing away over 2000
RandomNums=[zeros(m,numSims);RandomNums];

%if (1-sum(alpha)-sum(beta)-0.5*sum(gamma))>0
%    UncondStd =  sqrt(omega/(1-sum(alpha)-sum(beta)-0.5*sum(gamma)));
%else %non stationary
%    UncondStd=1;
%end

%h=UncondStd.^2*ones(t+m,1); % The first m values of h_t is to use the unconditional volatility
%data=UncondStd*ones(t+m,1);

h = ones(t+m,numSims); 
data = ones(t+m,numSims); 
h(1:m,:) = startVol;
data(1:m,:) = startResid;
T=size(data,1);
Idata=zeros(size(data));

parameters=[omega;alpha;gamma;beta];

for ii = 1:numSims
if tarch_type==1
    for t = (m + 1):T
        % As we move further than m, the value of h(t) will depends on the
        % parameters and random numbers.
        h(t,ii) = parameters' * [1 ; abs(data(t-(1:p),ii));  Idata(t-(1:o),ii).*abs(data(t-(1:o),ii)); h(t-(1:q),ii)];
        data(t,ii)=RandomNums(t,ii)*h(t,ii);
        Idata(t,ii)=data(t,ii)<0;
    end
    h(:,ii)=h(:,ii).^2;
else
    for t = (m + 1):T
        h(t,ii) = parameters' * [1 ; data(t-(1:p),ii).^2;  Idata(t-(1:o),ii).*data(t-(1:o),ii).^2; h(t-(1:q),ii)];
        data(t,ii)=RandomNums(t,ii)*sqrt(h(t,ii));
        Idata(t,ii)=data(t,ii)<0;
    end
end
end
%simulatedata=data((m+1+2000):T);
%ht=h(m+1+2000:T);
simulatedata=data(m+1:T,:);
ht=h(m+1:T,:);
%simulatedata = mean(simulatedataAll,2); 
%ht = mean(htAll,2);