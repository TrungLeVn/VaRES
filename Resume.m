function outNew= Resume(Windows,estParsOld,exitOld,model,submodel,period,nlags)
% Define some input
quantile = [0.01,0.025,0.05];
N = size(Windows,2); 
% First check converge
check = sum(exitOld,2); 
whichQuant = find(check ~= N);
estParsNew = estParsOld;
exitNew = exitOld;
for i = 1:length(whichQuant)
    whichWindows = find(exitOld(whichQuant(i),:) ~= 1);
    estParsNewQuant = estParsOld{1,whichQuant(i)};
    exitNewQuant = exitOld(whichQuant(i),:);
    for ii = 1:length(whichWindows)
        for rep = 1:2
        %[newPars,newExit] = GetFor(Windows{1,whichWindows(ii)},model,submodel,...
        %quantile(whichQuant(i)),period,nlags,estParsNewQuant(:,whichWindows(ii)));
        %%
        % Turn on if using existing estPars doesn't help. Will reestimate
        % the model completely
        [newPars,newExit] = GetFor(Windows{1,whichWindows(ii)},model,submodel,...
        quantile(whichQuant(i)),period,nlags);
        %%
        if newExit == 1
            'ok'
            break
        end
        end
    estParsNewQuant(:,whichWindows(ii)) = newPars;
    exitNewQuant(1,whichWindows(ii)) = newExit;
    end
    estParsNew{1,whichQuant(i)} = estParsNewQuant;
    exitNew(whichQuant(i),:) = exitNewQuant;
end
outNew.estParsNew = estParsNew;
outNew.exitNew = exitNew;
end