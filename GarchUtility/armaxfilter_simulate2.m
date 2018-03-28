function ySim = armaxfilter_simulate2(errors,const,ar,ARparams,ARlags,ma,MAparams,MAlags,X,Xparams)
% ARMAX(P,Q) simulation with normal errors.  Also simulates AR, MA and ARMA models.
%
% USAGE:
%   AR:
%   Y = armaxfilter_simulate2(e,CONST,AR,ARPARAMS,ARlags)
%   MA:
%   Y = armaxfilter_simulate2(e,CONST,0,[],MA,MAPARAMS,MAlags)
%   ARMA:
%   Y = armaxfilter_simulate2(T,CONST,AR,ARPARAMS,ARlags,MA,MAPARAMS,MAlags);
%   ARMAX:
%   Y = armaxfilter_simulate2(T,CONST,AR,ARPARAMS,ARlags,MA,MAPARAMS,MAlags,X,XPARAMS);
%
% INPUTS:
%   errors   - T by numSims matrix of simulated errors - Result of simulation on the GARCH
%              specification.
%   CONST    - Value of the constant in the model.  To omit, set to 0.
%   AR       - Order of AR in model.  To include only selected lags, for example t-1 and t-3, use 3
%                and set the coefficient on 2 to 0 
%   ARPARAMS - AR by 1 vector of parameters for the AR portion of the model
%   ARlags   - AR by 1  vector of AR lags. Use to initiate the
%              simulation
%   MA       - Order of MA in model.  To include only selected lags of the error, for example t-1
%                and t-3, use 3 and set the coefficient on 2 to 0 
%   MAPARAMS - MA by 1 vector of parameters for the MA portion of the model
%   MAlags   - MA by 1 vector of MA lags. Use to initiate the simulation
%   X        - T by K matrix of exogenous variables
%   XPARAMS  - K by 1 vector of parameters on the exogenous variables
%
% OUTPUTS:
%   Y        - A T by 1 vector of simulated data
%
% COMMENTS:
%   The ARMAX(P,Q) model simulated is:
%      y(t) = const + arp(1)*y(t-1) + arp(2)*y(t-2) + ... + arp(P) y(t-P) +
%                   + ma(1)*e(t-1)  + ma(2)*e(t-2)  + ... + ma(Q) e(t-Q)
%                   + xp(1)*x(t,1)  + xp(2)*x(t,2)  + ... + xp(K)x(t,K)
%                   + e(t)
% See also ARMAXFILTER, HETEROGENEOUSAR

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 3    Date: 9/1/2005

% Modified by Trung Le to allow for the simulation based simulated errors
% from GARCH simulation and supplied of ARlags and MAlags
% Modification: 1   Date: 08/02/2018

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch nargin
    case 2
        ar=0;
        ma=0;
        ARparams=[];
        MAparams=[];
        ARlags = []; 
        MAlags = [];
        X=[];
        Xparams=[];
    case 5
        ma=0;
        MAparams=[];
        MAlags = [];
        X=[];
        Xparams=[];
    case 8
        X=[];
        Xparams=[];
    case 10
        % Nothing
    otherwise
        error('The number of inputs must be 2, 5, 8 or 10.')
end

T=size(errors,1);

if ~isempty(ar) && (length(ARparams)<ar || min(size(ARparams))>1)
    error('Incorrect number of AR parameters')
end

if size(ARparams,2)<size(ARparams,1)
    ARparams=ARparams';
end

if length(ARlags) ~= ar
    error('ARlags need to at the same length as the AR order')
end

if ~isempty(ma) && (length(MAparams)<ma || min(size(MAparams))>1)
    error('Incorrect number of MA parameters')
end

if size(MAparams,2)<size(MAparams,1)
    MAparams=MAparams';
end

if length(MAlags) ~= ma
    error('MAlags need to at the same length as the MA order')
end

if length(Xparams)<size(X,2)
    error('Incorrect number of X parameters.  XPARAMS should be K by 1.')
end

if ~isempty(X) && length(X)~=T
    error('X should be length T');
end

if any([ma,ar]<0) || any([length(ma) length(ar)]>1)
    error('MA and AR must all be non negative scalars.')
end

if ~isscalar(const)
    error('CONST must be a scalar')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(X)
    X=0;
    Xparams=0;
else
    if size(Xparams,2)<size(Xparams,1)
        Xparams=Xparams';
    end
end

% Total length of the simulated data depends on the order of arma and the
% requied simulated length
m = max(ar,ma);
Tfull = size(errors,1) + m;
numSims = size(errors,2);
%Easy to set up the exogenous variables: If does not provided, just be the
%constant
ySim = zeros(T,numSims);
for ii = 1:numSims
    e = errors(:,ii);
    meanX=mean(X);
    exog=[repmat(meanX,length(e)-length(X),1)*Xparams';
    X*Xparams']+const;

if ma>0
    e=[MAlags; e];
    [e,elag]=newlagmatrix(e,ma,0);
    exog=exog+elag*MAparams'+e;    
else
    elag=0;
    exog=exog+e;
end

% Organize the data with regards to the Tfull 
exog = [zeros(Tfull - length(exog),1);exog];

% The starting values for y depends on the AR order and ARlags provided
y = zeros(Tfull,1);
y(Tfull - T:-1:Tfull - T - ar + 1) = ARlags;

if ar>0
    for i=ar+1:Tfull
        y(i)=exog(i)+ARparams*y(i-1:-1:i-ar);
    end
else
    y=exog;
end

%Fix the size
ySim(:,ii)=y(m+1:Tfull);
end
end