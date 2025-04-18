

%% ------------------------------------------------------------------------
% Parameters:

numTrials                = 480;
Mu                       = nan(1,numTrials); % outcome means
Haz                      = 0.1; % Hazard Rate

% Neural Network:
contextMeans             = -pi:0.01:pi;% "context layer" with VM tuned neurons
contextConc              = 32;%PDF concentration
contextStartValue        = 0;
outputMeans              = -50:10:350;% "Output Layer" with Gaussian units
fixedLR                  = 0.1;
FG                       = 0.1; % multiplier by which unused context weights are scaled for forgetting
actThresh                = 0.01; % threshold for forgetting weights
useLinearReadout         = true; % Readout of context layer. Default: True
shiftIncs                = linspace(0,2,50);

%% ------------------------------------------------------------------------
% Simulate:
load('simulatedLearningRateCP.mat');
load('McGuireNassar2014data.mat')
outcome = allDataStruct.currentOutcome(allDataStruct.subjNum==1);
noiseStd = allDataStruct.currentStd(allDataStruct.subjNum==1);
outputStd = noiseStd;

B = nan(size(outcome))'; pCha = B; R = pCha;
% Use normative model to compute CPP, RU & (model)LR:
for i=1:4
    [B1, totSig, R1, pCha1, ~] = frugFun5(outcome((i-1)*120+1:i*120)', Haz, noiseStd(i*120), 0, 1, false...
        , 150, .1);
    R(1,(i-1)*120+1:i*120)= R1(1:end-1);
    pCha(1,(i-1)*120+1:i*120)= pCha1;
    B(1,(i-1)*120+1:i*120)=B1(1:end-1);
end

RU=1./(R+1);
modelLR=RU+pCha - pCha.*RU;


% Initialize random weights matrix:
weightMatrix =(rand(length(outputMeans),...
    length(contextMeans))-.5)./1000; % currently initializing to positive weights -- should try balanced weights too.

% Initialize estimates of the network:
estimate=nan(size(outcome));
contextValue= contextStartValue;

% Initialize context layer to fire:
contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
contextAct=contextAct./sum(contextAct);


%Loop through trials to get network responses:
for i = 1:length(outcome)
    
    % Get summed inputs for output layer:
    outputAct=contextAct*weightMatrix'; % Produce a response
    outputAct(outputAct<0)=0; % Currently the only non-linearity is to get rid of negative firing rates
    outputAct=outputAct./(sum(outputAct));
    
    % Linear readout of output neurons:
    if   useLinearReadout
        estimate(i)=outputMeans* (outputAct)'; % Normalize activity...
    else %   Readout maximum
        estimate(i)=outputMeans(find(outputAct==max(outputAct),1));
        
    end
    
    
    contextShift(i)=LR2shift(shiftIncs',simulatedLearningRateCP,modelLR(i));
    
    % Shift the context value :
    contextValue=contextValue+contextShift(i);
    
    % Set context layer activity based on new context value.
    contextAct=circ_vmpdf(contextValue, contextMeans, contextConc);
    contextAct=contextAct./sum(contextAct);
    
    % Store dynamic context shift model context for dissimilarity matrix:
    x=size(contextMeans,2)*floor(contextValue/(2.*pi));
    totalContext(i,x+1:x+size(contextMeans,2))= contextAct;
    
    % Provide supervised signal for learning:
    targetActivation=normpdf(outcome(i), outputMeans, outputStd(i)); % we're giving model some info about noise
    targetActivation=targetActivation./(sum(targetActivation));
    normAdjusts=(targetActivation)'*contextAct;
    
    % Update weights:
    weightMatrix=weightMatrix.*(1-fixedLR) +fixedLR.* normAdjusts;
    
    % Forget inactive weights:
    weightMatrix(:, contextAct<actThresh)=...
        weightMatrix(:,contextAct<actThresh).*FG;
    
end
    
    
