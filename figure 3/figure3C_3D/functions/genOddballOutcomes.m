% Script to generate some "oddball" outcomes.


% task parameters:

numTrials= 1000; % number of trials;
driftRate= 5; % standard deviation of random walk in heli/cannon position
haz      = .1; % hazard rate of oddball event
noise    = 25;

% KalmanGain = driftRate^2 ./(driftRate^2 + noise^2)


% preallocate space for variables:
outcome=nan(numTrials, 1);
Mu     =nan(numTrials, 1);
isOdd  =nan(numTrials, 1);

% Loop through trials and generate outcomes:
for i = 1:numTrials
    
    % Choose mean:
    if i ==1
       Mu(i)=rand.*100+ 100; % start helicopter in the middlish region
    else 
       Mu(i)=Mu(i-1)+normrnd(0, driftRate);
    end
    
    % make boundary reflective so we don't get caught on edge
    if Mu(i)>300
        Mu(i) = 300 -  (Mu(i)-300); % reflective edges -- if you go past you bounce back. 
    elseif Mu(i)<0
        Mu(i)=abs(Mu(i));
    end
    
    % Choose outcome
    if rand>haz
        outcome(i)=min([max([normrnd(Mu(i), noise), 0]), 300]); % choose outcome from distribution centered on mean
        isOdd(i)=false;
    else
        outcome(i)=rand.*300; % choose outcome from distribution centered on mean
        isOdd(i)=true;
    end

    
    
    
    
end
isOdd=logical(isOdd);

xVar=1:numTrials
hold on
plot(xVar, outcome, 'o')
plot(xVar(isOdd), outcome(isOdd), 'or')

plot(1:numTrials, Mu, '-')



%% run reduced bayesian model over outcomes:



% Now lets make normative predictions for those trials


pred=nan(numTrials, 1);
pred(1)=150;

for i = 1:numTrials
    
    
    if i ==1
    unc(i)=.9;
    else 
    unc(i)=errBased_RU(2);  
    end   
    
    % Step 1 = compute prediction error...
    PE=outcome(i)-pred(i); %%
    
    % Step 2 = compute surprise and uncertainty according to prediction
    % error

    [errBased_pOdd, errBased_RU, errBased_LR, errBased_UP]=getTrialVarsFromPEs_cannon...
    ([noise; noise], [PE; 0], [haz], [1; 0], [false, false], unc(i), 0, 1, [1], [driftRate; driftRate] , [true; true], 300)

    % Step 3 = Store some stuff:
    oddProb(i)=errBased_pOdd(1);
   
    
    % Step 4 = update prediction
    pred(i+1)=pred(i)+errBased_UP(1);
    
    
    
end




hold on

xVar=1:numTrials
hold on
plot(xVar, outcome, 'o')
plot(xVar(isOdd), outcome(isOdd), 'or')
plot(xVar,    pred(1:end-1), 'c')
plot(1:numTrials, Mu, '-')






