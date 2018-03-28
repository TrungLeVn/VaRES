function [VaR,ES] = VaRhist(data, windowsize, theta, period, date)
%%
% This code is to compute the historical VaR and ES 
% INPUT:
% data              - The Tx1 column raw data; 
% windowsize        - The window size to estimate simulation. At least 100
% quantile          - The probability level of interest belongs to (0,1);
% period            - Return horizon
% date              - The Tx1 datenum column
%%
% INPUT CHECKING
N = size(data,1); 
if nargin < 5
    date = 1:N;
end

if any(theta) <= 0 || any(theta) > 1
    error('Quantile level need to be between 0 and 1'); 
end

if windowsize < 100 
    error('Window size should be at least 100 observations'); 
end
        
% Compute the VaR and ES based on window size
maxEnd = 10*floor(N/10);  %because we build the window by moving forward 10 observations
WindowEnd = windowsize : period: maxEnd-1; 
numHist = length(WindowEnd); 
VaR = zeros(numHist,length(theta)); 
ES = zeros(numHist,length(theta));
Date = zeros(numHist,1);
for ii = 1:length(theta)
for i = 1:numHist
    Windowdata = data(WindowEnd(i)-windowsize+1:WindowEnd(i)); 
    Windowdata = sort(Windowdata,1);
    tempVaR = Windowdata(round(theta(ii)*windowsize));
    %tempVaR = quantile(Windowdata,theta(ii));
    tempES = mean(Windowdata(Windowdata <= tempVaR));
    VaR(i,ii) = sqrt(period)*tempVaR; 
    ES(i,ii) = sqrt(period)*tempES;
    if ii == 1
    Date(i,ii) = date(WindowEnd(i)+1);
    end
end
end
VaR = [Date,VaR]; ES = [Date,ES];
end