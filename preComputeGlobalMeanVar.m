function preComputeGlobalMeanVar(lexicon)
%% preComputeGlobalMeanVar
global obsMat
list = who(obsMat);
for word = lexicon
   fprintf('Begin preComputeGlobalMeanVar for %s...\n', word);
   nTrain = length(list);
   allObs = []; % concatenation of all Observations
   for iUtterance = 1:nTrain % training utterances
      % global mean/var
      obsName = list{iUtterance};
      fprintf('\t\tinitialize_word %d of %d \t %s\n', iUtterance, nTrain, obsName);
      observation = obsMat.(obsName).';
      allObs = [allObs, observation];
   end
end

globalMu = mean(allObs.').';    globalVar = var(allObs.').';
save globalMuVar globalMu globalVar
