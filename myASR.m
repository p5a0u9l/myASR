function myASR(varargin)
%% myASR

%% Initialization and Setup
global debug p featureRoot trainRoot myVocab testLexicon fs
global nTestWord nTrainWord obsMat wavMat trainLexicon
debug = 1;
wavMat = matfile('TIDIG'); tic;
nTrainWord = 150;
nTestWord = 100;

p = inputParser;
addParameter(p, 'HMMTRAIN', 0);
addParameter(p, 'HMMTEST', 0);
addParameter(p, 'RECOGNITION', 1);
parse(p, varargin{:});

fs = 8e3;
testLexicon = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
trainLexicon = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
myVocab = {'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'}';
% myVocab = {'Zero', 'One', 'Two', 'Three', 'Four'}';

trainRoot = 'myTrained';
featureRoot = 'myFeatures';

%% Main 
if and(p.Results.HMMTRAIN, ~p.Results.HMMTEST) % Generic training
   M = 8; 
   hmmTraining(trainLexicon, M);
end

if p.Results.HMMTEST % Advanced training
   mIter = 1; 
   WER = ones(mIter, length(trainLexicon));
   m = (1:mIter) + 7;
   
   for i = 1:mIter
      if p.Results.HMMTRAIN
         fprintf('Begin training for M = %d\n', m(i));
         hmmTraining(trainLexicon, m(i));
      end
      WER(i, :) = hmmTesting(testLexicon, m(i));
   end
   [~, mOptimal] = min(WER);
   states = m(mOptimal);
   save([featureRoot, filesep, 'optimalStates'], 'states');
end

if p.Results.RECOGNITION
   interactiveRecognition()
end

%% Subfunctions

function hmmTraining(trainLexicon, M)
%%    HMMTRAINING
global debug featureRoot nTrainWord trainRoot baseline
% Initialize training files
if debug; fprintf('\tInitializing for word set %s... ', trainLexicon);  end;
obsMat = matfile([featureRoot, filesep, 'train_features_isolated_digits']);
list = who(obsMat);toc
preComputeMfcc = 0;
preComputeMuSigma = 0;

% Initial Parameters
if preComputeMfcc; preComputeTrainMfcc(list); end; % else saved to mat file
if preComputeMuSigma; preComputeGlobalMeanVar(trainLexicon);  end;

a_ij0 = mk_leftright_transmat(M, 0.6);
prior0 = normalise(rand(M, 1));

% Lexicon Loop
for iWord = 1:length(trainLexicon)
   % set up training data
   word = trainLexicon(iWord);
   if debug; fprintf('\tLoading training utterances for word %s...', word);  end;
   wordList = filterList(list, word);
   data = cell(nTrainWord, 1);
   for i = 1:nTrainWord
      data{i} = obsMat.(wordList{i}).';
   end;toc
   
   % initial Gaussian
   load([featureRoot, filesep, 'globalMuVar'])
   sigma1 = repmat(globalVar, [1, M]);
   mu1 = repmat(globalMu, [1, M]);
   
   % EM Training
   [prior, a_ij, mu1, sigma1] = myHmmTrain(data, prior0, a_ij0, mu1, sigma1);
   
   % Save Result
   saveName = fullfile(trainRoot, sprintf('trainedHmm_%s_M%d', word, M));
   save(saveName, 'prior', 'a_ij', 'mu1', 'sigma1');
end

function WER = hmmTesting(testLexicon, M)
%%    HMMTESTING
global wavMat nTestWord debug p fs
list = who(wavMat);
for iWord = 1:length(testLexicon)
   word = testLexicon(iWord);
   if debug; fprintf('\tBeing testing word %s... ', word);  end;
   wordList = filterList(list, word);
   wordList = wordList(1:min(nTestWord, length(wordList)));
   
   nCorrect = 0;
   for iTest = 1:length(wordList)
      % Load Test Audio
      testName = wordList{randi(length(wordList))};
      s = wavMat.(testName);
      if 0, fprintf('\t\tThe test utterance is %s...\n', testName); end
      
      % select w that maximizes pr(x|w, lambda)
      LogLikelihood = recognizeSpeech(s, M*ones(length(testLexicon), 1));
      [~, idx] = max(LogLikelihood);
      
      % compute WER
      testWord = testName(end-1);
      gestWord = testLexicon(idx);
      nCorrect = nCorrect + (strcmp(testWord, gestWord));
   end
   WER(iWord) = 1 - nCorrect/nTestWord;
   fprintf('\tiWord: %s, WER: %.2f Percent\n', word, WER(iWord)*100);
end

function LogLikelihood = recognizeSpeech(s, states)
%%    RECOGNIZESPEECH
global featureRoot trainRoot testLexicon fs p
if nargin < 2
   load([featureRoot, filesep, 'optimalStates']);
   if length(states) < length(testLexicon)
      states = repmat(states(1), [1, length(testLexicon)]);
   end
end
% Compute cepstral coefficients and delta coefficients
[obs, ~] = melcepst(s, fs, '0d', 12, 26, round(fs*0.025), ...
   round(fs*0.01), 300/fs, 3700/fs);
obs = obs.';

% Iterate over each word, load its trained HMM, compute pr(x|w, lambda)
LogLikelihood = zeros(length(testLexicon), 1);
for iWord = 1:length(testLexicon)
   word = testLexicon(iWord);
   M = states(iWord);
   hmmName = fullfile(trainRoot, sprintf('trainedHmm_%s_M%d', word, M));
%    if p.Results.RECOGNITION, fprintf('\t\tLoading %s...\n', hmmName); end
   load(hmmName)
   B = obsProbs(obs, mu1, sigma1);
   [~, ~, ~, LogLikelihood(iWord), ~] = forwardbackward(prior, a_ij, B);
end

function interactiveRecognition()
%%    INTERACTIVERECOGNITION
global myVocab fs
again = true;
iTest = 0;
nCorrect = 0;

while again
   iTest = iTest + 1;
   % cli intro
   if iTest == 1
      fprintf('\nWelcome to my simple ASR routine!\n'); tic;
      fprintf('My vocabulary is ');
      for i = 1:length(myVocab)
         fprintf('%s, ', myVocab{i})
      end; fprintf('\n');
      fprintf('Speak at the prompt to have single-digit speech recognized...\n')
      fprintf('Press Enter when you are ready\n\n'); pause;
   end
   
   % Capture New Audio
   s = record_voice(2.5, 1);
   s = condition_voice(s, fs);
   
   % select w that maximizes pr(x|w, lambda)
   LogLikelihood = recognizeSpeech(s);
   [val, idx] = max(LogLikelihood);
   
   % final outputs
   resultTable = sortrows(table(myVocab, LogLikelihood), 2, 'descend')
   soundsc(s, fs);
   fprintf('\nI think you said %s.\n', myVocab{idx});
   again = input('\nGo again? (1, 0): ');
end
