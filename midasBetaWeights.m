function weights = midasBetaWeights(nlag,param1,param2)
seq = linspace(eps,1-eps,nlag);
if param1 == 1    
    weights = (1-seq).^(param2-1);    
else
    weights = (1-seq).^(param2-1) .* seq.^(param1-1);    
end
weights = weights ./ nansum(weights);
end
