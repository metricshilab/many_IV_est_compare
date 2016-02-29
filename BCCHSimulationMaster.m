nRep = 5;
n = [100;250];
nn = size(n,1);
p = 100;
pnz = 5;
indp = (0:p-1)';
Fstat = [30;180];
nF = size(Fstat,1);
s2e = 1;
Cev = .6;
s2z = 1;
szz = .5;
pi1 = .7;
alpha = 1;

% Common design elements
SZ = s2z*toeplitz((szz).^indp);
cSZ = chol(SZ);
cFS = [ones(pnz,1);zeros(p-pnz,1)];
scale = zeros(nF,nn);
s2v = zeros(nF,nn);
for ii = 1:nn
    for jj = 1:nF
        scale(jj,ii) = sqrt(Fstat(jj)/((Fstat(jj)+n(ii))*cFS'*SZ*cFS));
        s2v(jj,ii) = 1-(scale(jj,ii)^2)*cFS'*SZ*cFS;
    end
end
sev = Cev*sqrt(s2e)*sqrt(s2v);

% Initialize matrices for conventional estimators
b2sls = zeros(nRep,nn,nF);
bfull = zeros(nRep,nn,nF);
s2sls = zeros(nRep,nn,nF);
sfull = zeros(nRep,nn,nF);
% Initialize matrices for IV with highest corr Z
b2sls1 = zeros(nRep,nn,nF);
s2sls1 = zeros(nRep,nn,nF);
% Initialize matrices for IV-LASSO estimators
FS = zeros(nRep,nn,nF);
blassoC = zeros(nRep,nn,nF);
blassoCF = zeros(nRep,nn,nF);
slassoC = zeros(nRep,nn,nF);
slassoCF = zeros(nRep,nn,nF);
% Initialize matrices for sup-score test inference
aTest = (.5:.01:1.5)';
na = size(aTest,1);
supScore = zeros(nRep,na,nn,nF);
supScore05 = zeros(nRep,na,nn,nF);
lambdaSS05 = norminv(1-.05/(2*p));
% Initialize matrix for CV Ridge penalty value
LambdaRidgeA = zeros(nRep,nn,nF);
LambdaRidgeB = zeros(nRep,nn,nF);
% Initialize matrices for IV with highest corr Z including ridge
Rb2sls1A = zeros(nRep,nn,nF);
Rs2sls1A = zeros(nRep,nn,nF);
Rb2sls1B = zeros(nRep,nn,nF);
Rs2sls1B = zeros(nRep,nn,nF);
% Initialize matrices for IV-LASSO estimators including ridge
RFSA = zeros(nRep,nn,nF);
RblassoCA = zeros(nRep,nn,nF);
RblassoCFA = zeros(nRep,nn,nF);
RslassoCA = zeros(nRep,nn,nF);
RslassoCFA = zeros(nRep,nn,nF);
RFSB = zeros(nRep,nn,nF);
RblassoCB = zeros(nRep,nn,nF);
RblassoCFB = zeros(nRep,nn,nF);
RslassoCB = zeros(nRep,nn,nF);
RslassoCFB = zeros(nRep,nn,nF);
RFSC = zeros(nRep,nn,nF);
RblassoCC = zeros(nRep,nn,nF);
RblassoCFC = zeros(nRep,nn,nF);
RslassoCC = zeros(nRep,nn,nF);
RslassoCFC = zeros(nRep,nn,nF);
% Initialize matrices for sup-score test inference including ridge
RaTestA = (.5:.01:1.5)';
RnaA = size(RaTestA,1);
RsupScoreA = zeros(nRep,RnaA,nn,nF);
RsupScore05A = zeros(nRep,RnaA,nn,nF);
RaTestB = (.5:.01:1.5)';
RnaB = size(RaTestB,1);
RsupScoreB = zeros(nRep,RnaB,nn,nF);
RsupScore05B = zeros(nRep,RnaB,nn,nF);
RlambdaSS05 = norminv(1-.05/(2*(p+1)));
%%

for ii = 1:nn
    for jj = 1:nF
        disp([n(ii) Fstat(jj)]);
        nUse = n(ii);
        SU = [s2e sev(jj,ii) ; sev(jj,ii) s2v(jj,ii)];
        cSU = chol(SU);
        lambda0C = 2.2*sqrt(2*log(2*p*log(nUse)/.1));
        Rlambda0C = 2.2*sqrt(2*log(2*(p+1)*log(nUse/2)/.1));
                        
        % Calculate all the estimators, etc. inside this loop
        for kk = 1:nRep
            if floor((kk-1)/10) == (kk-1)/10
                disp(kk)
            end
            ZOrig = randn(nUse,p)*cSZ;
            U = randn(nUse,2)*cSU;
            xOrig = scale(jj,ii)*ZOrig*cFS+U(:,2);
            yOrig = alpha*xOrig+U(:,1);
            Z = ZOrig - ones(nUse,1)*mean(ZOrig);
            x = xOrig - mean(xOrig);
            y = yOrig - mean(yOrig);
          
            % LASSO estimators
            e0 = x;
            Ups0 = sqrt((e0.^2)'*(Z.^2))';
            coefTemp = LassoShooting2(Z,x,lambda0C*Ups0,'verbose',0);
            ind0 = (abs(coefTemp) > 0);
            Z0 = Z(:,ind0);
            for mm = 1:15
                e1 = x-Z0*(Z0\x);
                Ups1 = sqrt((e1.^2)'*(Z.^2))';
                coefTemp = LassoShooting2(Z,x,lambda0C*Ups1,'verbose',0);
                ind1 = (abs(coefTemp) > 0);
                Z0 = Z(:,ind1);
            end                
            Z1 = Z(:,ind1);
                        
            if isempty(Z1),
                blassoC(kk,ii,jj) = NaN;
                slassoC(kk,ii,jj) = NaN;
                blassoCF(kk,ii,jj) = NaN;
                slassoCF(kk,ii,jj) = NaN;
                FS(kk,ii,jj) = 0;
            else
                bfs = Z1\x;
                efs = x - Z1*bfs;
                Vfs = ((Z1'*Z1)\((Z1.*((efs.^2)*ones(1,sum(ind1))))'*Z1))/(Z1'*Z1);
                FS(kk,ii,jj) = bfs'*(Vfs\bfs)/sum(ind1);
                [btemp1,VCtemp1] = tsls(y,x,[],Z1);
                [btemp2,~,~,VCtemp2] = fuller(y,x,[],Z1,1);
                blassoC(kk,ii,jj) = btemp1(1);
                slassoC(kk,ii,jj) = sqrt(VCtemp1(1,1));
                blassoCF(kk,ii,jj) = btemp2(1);
                slassoCF(kk,ii,jj) = sqrt(VCtemp2(1,1));
            end
            
            
            % Sup-Score Tests
            for mm = 1:na
                aEval = aTest(mm,1);
                eTemp = y-aEval*x;
                ScoreVec = eTemp'*Z;
                ScoreStd = sqrt((eTemp.^2)'*(Z.^2));
                ScaledScore = ScoreVec./(1.1*ScoreStd);
                supScore05(kk,mm,ii,jj) = max(abs(ScaledScore)) > lambdaSS05;
                supScore(kk,mm,ii,jj) = max(abs(ScaledScore));
            end
            
            % Calculate estimators putting in ridge fit with coefficients 
            % estimated from 1/2 of sample
            
            % Split sample
            UseRidgeB = (1:nUse)';
            UseRidgeA = randsample(nUse,floor(nUse/2));
            nRidgeA = size(UseRidgeA,1);
            UseRidgeB(UseRidgeA,:) = [];
            nRidgeB = size(UseRidgeB,1);
            
            yA = yOrig(UseRidgeA);
            xA = xOrig(UseRidgeA);
            ZA = ZOrig(UseRidgeA,:);
            yB = yOrig(UseRidgeB);
            xB = xOrig(UseRidgeB);
            ZB = ZOrig(UseRidgeB,:);
            
            try
                LambdaRidgeA(kk,ii,jj) = ...
                    fminunc(@(z) CVRidge(z,xA,ZA),...
                    p*s2v(jj,ii)/(50*scale(jj,ii)^2),...
                    optimset('disp','off','MaxFunEvals',1000));
            catch %#ok<CTCH>
                LambdaRidgeA(kk,ii,jj) = ...
                    fmincon(@(z) CVRidge(z,xA,ZA),...
                    p*s2v(jj,ii)/(50*scale(jj,ii)^2),...
                    [],[],[],[],.1,1e7,[],...
                    optimset('disp','off','MaxFunEvals',1000));
            end
            try
                LambdaRidgeB(kk,ii,jj) = ...
                    fminunc(@(z) CVRidge(z,xB,ZB),...
                    p*s2v(jj,ii)/(50*scale(jj,ii)^2),...
                    optimset('disp','off','MaxFunEvals',1000));
            catch %#ok<CTCH>
                LambdaRidgeB(kk,ii,jj) = ...
                    fmincon(@(z) CVRidge(z,xB,ZB),...
                    p*s2v(jj,ii)/(50*scale(jj,ii)^2),...
                    [],[],[],[],.1,1e7,[],...
                    optimset('disp','off','MaxFunEvals',1000));
            end

            
            RidgeFitB = [ones(size(yB)) ZB]*ridge(xA,ZA,...
                LambdaRidgeA(kk,ii,jj),0);
            RidgeFitA = [ones(size(yA)) ZA]*ridge(xB,ZB,...
                LambdaRidgeB(kk,ii,jj),0);

            ZB = [ZB RidgeFitB]; %#ok<AGROW>
            ZA = [ZA RidgeFitA]; %#ok<AGROW>

            ZA = ZA - ones(nRidgeA,1)*mean(ZA);
            xA = xA - mean(xA);
            yA = yA - mean(yA);
            
            ZB = ZB - ones(nRidgeB,1)*mean(ZB);
            xB = xB - mean(xB);
            yB = yB - mean(yB);
            
                           
            % Highest correlation 2SLS
            ZL1A = ZA(:,corr(ZA,xA) == max(corr(ZA,xA)));
            [btemp1,VCtemp1] = tsls(yA,xA,[],ZL1A);

            Rb2sls1A(kk,ii,jj) = btemp1(1);
            Rs2sls1A(kk,ii,jj) = sqrt(VCtemp1(1,1));

            ZL1B = ZB(:,corr(ZB,xB) == max(corr(ZB,xB)));
            [btemp1,VCtemp1] = tsls(yB,xB,[],ZL1B);
            
            Rb2sls1B(kk,ii,jj) = btemp1(1);
            Rs2sls1B(kk,ii,jj) = sqrt(VCtemp1(1,1));
                        
            
            
            % LASSO estimators
            % A Sample
            e0 = xA;
            Ups0 = sqrt((e0.^2)'*(ZA.^2))';
            coefTemp = LassoShooting2(ZA,xA,Rlambda0C*Ups0,'verbose',0);
            ind0 = (abs(coefTemp) > 0);
            ZL0 = ZA(:,ind0);
            for mm = 1:15
                e1 = xA-ZL0*(ZL0\xA);
                Ups1 = sqrt((e1.^2)'*(ZA.^2))';
                coefTemp = LassoShooting2(ZA,xA,Rlambda0C*Ups1,'verbose',0);
                ind1 = (abs(coefTemp) > 0);
                ZL0 = ZA(:,ind1);
            end                
            ZL1A = ZA(:,ind1);
                        
            if isempty(ZL1A),
                RblassoCA(kk,ii,jj) = NaN;
                RslassoCA(kk,ii,jj) = NaN;
                RblassoCFA(kk,ii,jj) = NaN;
                RslassoCFA(kk,ii,jj) = NaN;
                RFSA(kk,ii,jj) = 0;
            else
                bfs = ZL1A\xA;
                efs = xA - ZL1A*bfs;
                Vfs = ((ZL1A'*ZL1A)\((ZL1A.*((efs.^2)*ones(1,sum(ind1))))'*ZL1A))/(ZL1A'*ZL1A);
                RFSA(kk,ii,jj) = bfs'*(Vfs\bfs)/sum(ind1);
                [btemp1,VCtemp1] = tsls(yA,xA,[],ZL1A);
                [btemp2,~,~,VCtemp2] = fuller(yA,xA,[],ZL1A,1);
                RblassoCA(kk,ii,jj) = btemp1(1);
                RslassoCA(kk,ii,jj) = sqrt(VCtemp1(1,1));
                RblassoCFA(kk,ii,jj) = btemp2(1);
                RslassoCFA(kk,ii,jj) = sqrt(VCtemp2(1,1));
                WA = RFSA(kk,ii,jj)*sum(ind1);
            end
            
            % B Sample
            e0 = xB;
            Ups0 = sqrt((e0.^2)'*(ZB.^2))';
            coefTemp = LassoShooting2(ZB,xB,Rlambda0C*Ups0,'verbose',0);
            ind0 = (abs(coefTemp) > 0);
            ZL0 = ZB(:,ind0);
            for mm = 1:15
                e1 = xB-ZL0*(ZL0\xB);
                Ups1 = sqrt((e1.^2)'*(ZB.^2))';
                coefTemp = LassoShooting2(ZB,xB,Rlambda0C*Ups1,'verbose',0);
                ind1 = (abs(coefTemp) > 0);
                ZL0 = ZB(:,ind1);
            end                
            ZL1B = ZB(:,ind1);
                        
            if isempty(ZL1B),
                RblassoCB(kk,ii,jj) = NaN;
                RslassoCB(kk,ii,jj) = NaN;
                RblassoCFB(kk,ii,jj) = NaN;
                RslassoCFB(kk,ii,jj) = NaN;
                RFSB(kk,ii,jj) = 0;
            else
                bfs = ZL1B\xB;
                efs = xB - ZL1B*bfs;
                Vfs = ((ZL1B'*ZL1B)\((ZL1B.*((efs.^2)*ones(1,sum(ind1))))'*ZL1B))/(ZL1B'*ZL1B);
                RFSB(kk,ii,jj) = bfs'*(Vfs\bfs)/sum(ind1);
                [btemp1,VCtemp1] = tsls(yB,xB,[],ZL1B);
                [btemp2,~,~,VCtemp2] = fuller(yB,xB,[],ZL1B,1);
                RblassoCB(kk,ii,jj) = btemp1(1);
                RslassoCB(kk,ii,jj) = sqrt(VCtemp1(1,1));
                RblassoCFB(kk,ii,jj) = btemp2(1);
                RslassoCFB(kk,ii,jj) = sqrt(VCtemp2(1,1));
                WB = RFSB(kk,ii,jj)*sum(ind1);
            end
            
            % Combine estimators
            if isempty(ZL1A) && isempty(ZL1B)
                RblassoCC(kk,ii,jj) = NaN;
                RslassoCC(kk,ii,jj) = NaN;
                RblassoCFC(kk,ii,jj) = NaN;
                RslassoCFC(kk,ii,jj) = NaN;
                RFSC(kk,ii,jj) = 0;
            elseif isempty(ZL1A) && ~isempty(ZL1B)
                RblassoCC(kk,ii,jj) = RblassoCB(kk,ii,jj);
                RslassoCC(kk,ii,jj) = RslassoCB(kk,ii,jj);
                RblassoCFC(kk,ii,jj) = RblassoCFB(kk,ii,jj);
                RslassoCFC(kk,ii,jj) = RslassoCFB(kk,ii,jj);
                RFSC(kk,ii,jj) = RFSB(kk,ii,jj);
            elseif ~isempty(ZL1A) && isempty(ZL1B)
                RblassoCC(kk,ii,jj) = RblassoCA(kk,ii,jj);
                RslassoCC(kk,ii,jj) = RslassoCA(kk,ii,jj);
                RblassoCFC(kk,ii,jj) = RblassoCFA(kk,ii,jj);
                RslassoCFC(kk,ii,jj) = RslassoCFA(kk,ii,jj);
                RFSC(kk,ii,jj) = RFSA(kk,ii,jj);
            else
                weightA = RslassoCB(kk,ii,jj)^2/...
                    (RslassoCA(kk,ii,jj)^2 + RslassoCB(kk,ii,jj)^2);
                weightB = RslassoCA(kk,ii,jj)^2/...
                    (RslassoCA(kk,ii,jj)^2 + RslassoCB(kk,ii,jj)^2);
                RblassoCC(kk,ii,jj) = weightA*RblassoCA(kk,ii,jj) + ...
                    weightB*RblassoCB(kk,ii,jj);
                RslassoCC(kk,ii,jj) = sqrt((weightA*RslassoCA(kk,ii,jj))^2 + ...
                    (weightB*RslassoCB(kk,ii,jj))^2);
                weightA = RslassoCFB(kk,ii,jj)^2/...
                    (RslassoCFA(kk,ii,jj)^2 + RslassoCFB(kk,ii,jj)^2);
                weightB = RslassoCFA(kk,ii,jj)^2/...
                    (RslassoCFA(kk,ii,jj)^2 + RslassoCFB(kk,ii,jj)^2);
                RblassoCFC(kk,ii,jj) = weightA*RblassoCFA(kk,ii,jj) + ...
                    weightB*RblassoCFB(kk,ii,jj);
                RslassoCFC(kk,ii,jj) = sqrt((weightA*RslassoCFA(kk,ii,jj))^2 + ...
                    (weightB*RslassoCFB(kk,ii,jj))^2);
                RFSC(kk,ii,jj) = (WA+WB)/(size(ZL1A,2)+size(ZL1B,2));
            end                
                
            
            
            % Sup-Score Tests
            for mm = 1:RnaA
                aEval = RaTestA(mm,1);
                eTemp = yA-aEval*xA;
                ScoreVec = eTemp'*ZA;
                ScoreStd = sqrt((eTemp.^2)'*(ZA.^2));
                ScaledScore = ScoreVec./(1.1*ScoreStd);
                RsupScore05A(kk,mm,ii,jj) = max(abs(ScaledScore)) > RlambdaSS05;
                RsupScoreA(kk,mm,ii,jj) = max(abs(ScaledScore));
                eTemp = yB-aEval*xB;
                ScoreVec = eTemp'*ZB;
                ScoreStd = sqrt((eTemp.^2)'*(ZB.^2));
                ScaledScore = ScoreVec./(1.1*ScoreStd);
                RsupScore05B(kk,mm,ii,jj) = max(abs(ScaledScore)) > RlambdaSS05;
                RsupScoreB(kk,mm,ii,jj) = max(abs(ScaledScore));
            end

                        
        end
    end
end
        
% Save output
save EMASimulation5Const0p6NewRidgeSplit.mat ;

