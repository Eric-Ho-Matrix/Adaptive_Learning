% CODE FROM MRN TO REMOVE JOYSTICK ERRORS


%% Most of this stuff is probably irrelevant -- but some of it applies (eg. minTrialPerBlock):
% analysis choices:
forceRun        = false;
zChangeThresh   = 2;     % z scored difference in means considered as identifiable change-point
sphereRad       = 3;
maxTAbC         = 7;
minTrialPerBlock= 3;
haz             = .1;    % hazard rate used for computing CPP and RU;
viewPlots       = false;
distanceMetric  = 1 ;    % 1 = 1-correlation (pearson), 2 = euclidian distance, 3 = 1- spearman correlation,
contextMetric   = 1 ;    % 1 = trial influence, 2 = migrating context, 3 = just previous trial learning rate, 4 = separate RU influenace and CP incluence (replaces std))
contextIndOffset= 0;     % 0 = align as if beliefs before outcome are represented, 1 of beliefs after outcome are represented
autoCorrStrategy= 1 ;    % 1 = diagonal autoregressors, 2 = migrating context autoregressor (1 term)
numRegs         = 12;    % for storage purposes... added subBlock regressor
numAutocorrTerms= 15;    % this only applies to the trial influence context
doSimpKarpova   = false; % this regresses pattern changes onto RU, CPP and RU
karpovaMod      = 2;  % mod 1 = CPP, RU, VAL; mod 2 = modLR



load('McGuireNassar2014data.mat')

% 1) get relevant behavioral measures:
% trials after obvious change-point
% helicopter location
% extra controls?
% prediction...
% bag location...
% continuous variables (ie model fit relative uncertainty)

heliLoc=allDataStruct.currentMean; % 
bagLoc=cat(1, allDataStruct.currentOutcome);
noise=cat(1, allDataStruct.currentStd);
prediction=cat(1, allDataStruct.currentPrediction);
tPerBlock=cat(1, allDataStruct.blockCompletedTrials);
isPredActive=cat(1, allDataStruct.isPredictionActive);

newBlock=tPerBlock<=1;
ln=length(newBlock);
blkNum=nan(ln,1);
blkNum(1)=1;
for j = 2:ln
    if tPerBlock(j)==1
        blkNum(j)=blkNum(j-1)+1;
    else
        blkNum(j)=blkNum(j-1);
    end
end

% tNum=1:length(heliLoc);
% % compute TAbC (trials after (big) change-point)
% TAbC=nan(size(heliLoc));
% sbNum=nan(size(heliLoc));
% for j = 1:length(heliLoc)
%     if newBlock(j) || (heliLoc(j)-heliLoc(j-1))./noise(j)>zChangeThresh
%         TAbC(j)=0;
%         if j==1
%             sbNum(j)=1;
%         else
%             sbNum(j)=sbNum(j-1)+1;
%         end
%         
%     else
%         TAbC(j)=TAbC(j-1)+1;
%         sbNum(j)=sbNum(j-1);
%     end
% end
% TAbC(TAbC>maxTAbC)=maxTAbC;
% % get coin info:
% coinDat=cat(1, subBehav.mrMain.statusData.trialData);
% coinType=cat(1, coinDat.coinType)  ;
% VAL =cat(1, coinType.value);

% Get Model based variables RU and CPP:
[~, UP, PE]=computeLR(bagLoc, prediction, newBlock)   ;
[errBased_pCha, errBased_RU]=getTrialVarsFromPEs(noise, PE, haz, false, [], newBlock, false(size(newBlock)), 1, 0);
% RUN basic behavioral model to get per-subject estimates of RU-driven
% learning:
num=1;
UP = UP(allDataStruct.subjNum==num);
PE = PE(allDataStruct.subjNum==num);
errBased_pCha = errBased_pCha(allDataStruct.subjNum==num);
errBased_RU = errBased_RU(allDataStruct.subjNum==num);

modLR    =errBased_pCha+errBased_RU - errBased_pCha.*errBased_RU;
% simulate behavior of model conditional on subject errors:
modSimBehav(1)=prediction(1);

for i = 1:(length(modLR)-1)
    if newBlock(i+1)==1
        modSimBehav(i+1)=prediction(i+1);
    else
        modPE=bagLoc(i)-modSimBehav(i);
        modUP=modPE*modLR(i);
        modSimBehav(i+1)=modSimBehav(i)+modUP;
    end
end

%Get simulated model behavior,
priorFrac=1-modLR;
nonCPfrac=1-errBased_pCha;
nonRUfrac=1-errBased_RU;
nonIntFrac=1- (errBased_pCha.*errBased_RU);
trialInfluence=nan(length(modLR), length(modLR));
CPP_based_influence=nan(length(modLR), length(modLR));
RU_based_influence=nan(length(modLR), length(modLR));
int_based_influence=nan(length(modLR), length(modLR));

ln=length(modLR);
migratingContext_dist=nan(ln);

for f = 1:ln
    for j = 1:ln
        % ok, influence of ith representation on jth trial:
        
        % 5-9-18 -- MRN fixed indexing so that we can now look at
        % things aligned to measure beliefs before outcome
        % (contextIndOffset==0)
        % Or to align things to measure beliefs after outcome
        % (contextIndOffset==1)
        
        if j>=f % MRN changed this to make trial influence matrix symmetrical 9/16/17
            trialInfluence(j,f)=prod(priorFrac(f+contextIndOffset:j-1+contextIndOffset));
        else
            trialInfluence(j,f)=prod(priorFrac(j+contextIndOffset:f-1+contextIndOffset));
        end
        
        CPP_based_influence(j,f)=prod(nonCPfrac(f:j-1));
        RU_based_influence(j,f)=prod(nonRUfrac(f:j-1));
        int_based_influence(j,f)=prod(nonIntFrac(f:j-1));
        migratingContext_dist(j,f)= sqrt(sum((modLR(f:j-1)).^2));
    end
end


% mean center each diagonal of the migrating context distance:
orthMigContext_dist=nan(ln);
for j = 1:ln
    LowT=tril(ones(ln), 1-j);
    LowerT=tril(ones(ln), -j);
    relDiag=LowT&~LowerT;
    orthMigContext_dist(relDiag)=migratingContext_dist(relDiag)- nanmean(migratingContext_dist(relDiag));
    % how much variance is in each diagonal?
    migContextVar(j)=nanstd( orthMigContext_dist(relDiag));
    trialInfluenceVar(j)=nanstd( trialInfluence(relDiag));
    metricSimilarity(j)=corr(orthMigContext_dist(relDiag), trialInfluence(relDiag));
end

%     hold on
%     plot((migContextVar), 'b')
%     plot((trialInfluenceVar).*2, 'r')
%     aa=legend('migrating context', 'trial influence')
%     set(aa,'box','off')
%     set(gca, 'box','off')
%     ylabel('variance')
%     xlabel('off diagonal')
%     saveas(gcf, 'howKP_metricsDiffer.eps','epsc2')
%     close all



% what do these things look like:
% keyboard

if viewPlots
    
    figure
    imagesc(CPP_based_influence, [0 1]);
    colorbar;
    
    figure
    imagesc(RU_based_influence, [0 1]);
    colorbar;
    
    figure
    imagesc(int_based_influence, [0 1]);
    colorbar;
    
    
    figure
    imagesc(trialInfluence, [0 1]);
    colorbar;
    
    
    
    % i'm not sure that i like the looks of the interaction. It gets a different time constant
    % just by the merit of having a different absolute value. This shouldn't
    % matter too much since we plan to include terms to deal with
    % autocorrelation though...
end





    






% create autocorrelation terms:

lt=tril(true(ln), -1); %lower triangle selector
autoCorrMat=nan(sum(lt(:)),ln-1); % matrix of all lower correlation diagonals
for j = 1:ln-1
    bt=tril(true(ln), (j)-ln);     % big triangle
    st=tril(true(ln), (j-1)-ln);   % little triangle
    diagTerm= bt&~st;              % diagonal
    autoCorrMat(:,j)=diagTerm(lt);
end


%% create good trial selector:
% I'm not sure that we need to ditch the trial after a big update in
% this analysis... but i'll leave to code around for now just in case.

bt=find(isPredActive==0);
bpt=(bt)+1;
allBT=false(ln, 1);
BT=false(ln, 1);
BPT=false(ln, 1);

allBT(bt)=true;
allBT(bpt)=true;
BT(bt)=true;     % nuisance
BPT(bpt)=true;   % nuisance

% Joystick errors: see fMRI_mastList.m for complete description.
tol=30; % this is the free parameter... higher tolerences identify fewer trials.
aMat=[nan nan; prediction(1:end-1) bagLoc(1:end-1)]; % this is a list of prev pred/outcomes
jsFail=~newBlock & ((prediction < (min(aMat, [], 2)-tol) |  prediction > (max(aMat, [], 2)+tol))); % we're way out of that range, call it a joystick fail.

% count purported failures
realJoystickFail=find(jsFail)-1; % backup failure to correct index
postJoystickFail=find(jsFail)-0; % this is the trial after failure
allJSF=false(ln, 1); % this is either of them.
all_postJSF=false(ln, 1);
allJSF(realJoystickFail)=true;
all_postJSF(postJoystickFail)=true;

% put all of these bad trials together in order to have a list of trials to
% omit for behavioral analaysis:
allBB=false(ln, 1);
allBB(allBT)=true;
allBB(realJoystickFail)=true;
allBB(postJoystickFail)=true;
allBB(tPerBlock<=minTrialPerBlock)=true; % lets toss the first trial of each block too.
        
save('trialInfluence','trialInfluence')
   