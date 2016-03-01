% ZT: fminsearch was used for optimization.
% check the dimension of his optimization
% it is single dimension, then does not work.


%%Regularized LIML for many instruments simulations Matlab code.
% This code compares the regulirized version of LIML with Carrasco 2012 
%regularized 2SLS and DN(2001). 
%%% Different regularized scheme are Tikhonv, Landweber Fridman, principal
%%% components, Donald and Newey and unfeasible IV estimator (denoted IV here)
%%The function obj_fun.m is need to run the code.
clear all
clc
tic
L=[15 30  50];
%format shortG
for t=1:1:length(L)

seed1=29384;
randn('seed',seed1);
n=500 ;	            %sample size
kbar=L(t);	        %maximum number of instruments
itn=1000;	        %replications for simulation 
kpc=kbar;             %principal components

gam=0.1;            %true value of gamma 
cov=0.5;            %covariance between u and e 
rfsqr=0.1;          %Rf square 
kv=zeros(itn,1);    %stores opt number of instruments in DN

d_t=zeros(itn,1);   v_t=zeros(itn,1);         %stores Tikhonov
d_l=zeros(itn,1);   v_l=zeros(itn,1);        %stores Landweber Fridman
d_pc=zeros(itn,1);  v_pc=zeros(itn,1);       %principal components
d_iv=zeros(itn,1);  v_iv=zeros(itn,1);       %IV with known instrument
d_sls=zeros(itn,1);  v_sls=zeros(itn,1);     %Donald and Newey
d_Li=zeros(itn,1);   v_Li=zeros(itn,1);      %LIML for IV with know instrument
d_tl=zeros(itn,1);   v_tl=zeros(itn,1);      %LIML for  Tikhonov
d_ll=zeros(itn,1);   v_ll=zeros(itn,1);      %LIML for LF
d_pcl=zeros(itn,1);   v_pcl=zeros(itn,1);      %LIML for PC
d_liml=zeros(itn,1);   v_liml=zeros(itn,1);      %LIML for DN
 V_Bt=zeros(itn,1);         %stores Tikhonov
   V_Bl=zeros(itn,1);        %stores Landweber Fridman
V_Bpc=zeros(itn,1);       %principal components
V_Biv=zeros(itn,1);       %IV with known instrument
V_Bsls=zeros(itn,1);     %Donald and Newey
V_BLi=zeros(itn,1);      %LIML for IV with know instrument
V_Btl=zeros(itn,1);      %LIML for  Tikhonov
V_Bll=zeros(itn,1);      %LIML for LF
V_Bpcl=zeros(itn,1);      %LIML for PC
V_Bliml=zeros(itn,1);      %LIML for DN
V_BLiml=zeros(itn,1);      %LIML 
v_BLiml=zeros(itn,1);      %LIML 
d_Liml=zeros(itn,1); 
nlf=300; % max iteration for Landweber Fridman 

aa=[0.01:0.01:0.5];% %Grid of values for alpha in Tikhonov
aa=aa';
nt=length(aa); % number of values for alpha in Tikhonov

tra=zeros(nt,2); trlf=zeros(nlf,2); trpc=zeros(kpc,2);
cvt=zeros(nt,1); cvlf=zeros(nlf,1); cvpc=zeros(kpc,1); cvsls=zeros(kbar,1);
st=zeros(nt, 1); slf=zeros(nlf,1); spc=zeros(kpc,1); ssls=zeros(kbar,1);
aopt=zeros(itn,1); lopt=zeros(itn,1); kopt=zeros(itn,1);


piv=sqrt(rfsqr/(kbar*(1-rfsqr)) )*ones(kbar,1); %same weight for all coefficients Model 1b.
 


for it=1:1:itn;
    %generate the simulated data 
kbar=L(t);
sigma=ones(2,2); %covariance matrix for e and u 
sigma(1,2)=cov; sigma(2,1)=cov;	
sigma=chol(sigma);

eu=sigma'*randn(2,n);	%simulated e and u matrix 

%vector of instruments 
x=randn(n,kbar); % X_i 
yy=x*piv+eu(2,:)';  % Y in Donald&Newey paper
y=gam*yy+eu(1,:)';
fv=x*piv;

%%%%%%Unfesable IV estimator%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

weight=pinv(fv'*fv); % weighting matrix 
d_iv(it)=(yy'*fv*weight*fv'*y)/(yy'*fv*weight*fv'*yy);
v_iv(it)=((y-d_iv(it)*yy)'*(y-d_iv(it)*yy)/n)/(yy'*fv*weight*fv'*yy);



[rLi(it), mu]=fminsearch(@(x)obj_fun(y,yy,fv*weight*fv',x),d_iv(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_Li(it)=(yy'*(fv*weight*fv'-mu*eye(n))*y)/(yy'*(fv*weight*fv'-mu*eye(n))*yy);
%d_Li(it)=rLi(it);
v_Li(it)=((y-d_Li(it)*yy)'*(y-d_Li(it)*yy)/n)/(yy'*(fv*weight*fv'-mu*eye(n))*yy);

%%%%LIML%%%
P=x*(x'*x)^(-1)*x';
d_il(it)=(yy'*P*y)/(yy'*P*yy);
[rLiml(it), mu]=fminsearch(@(x)obj_fun(y,yy,P,x),d_il(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_Liml(it)=(yy'*(P-mu*eye(n))*y)/(yy'*(P-mu*eye(n))*yy);


kn=x'*x/n;
[ve, va]=eig(kn);

va=sum(va);
va=va(kbar:-1:1); ve(:,1:1:kbar)=ve(:,kbar:-1:1);%Reverses the order 
va2=va.^2;
for k=1:1:kbar;
     psi(:,k)=x*ve(:,k)* sqrt(va(k))^(-1); %normalization 
 end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Tikhonov %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% preliminary estimator of sigma_u and sigma_ueps
 for a=1:1:nt;
vv=(va2+aa(a)).^(-1);

vc=repmat((va2.*vv),n,1);
Pa=(psi.*vc)*psi'/n; 
u=(eye(n)-Pa)*yy;
tra(a,:)=[sum(diag(Pa)) sum(diag(Pa*Pa))];
cvt(a)=(u'*u/n)/(1-sum(diag(Pa))/n)^2;
 end

[min1, a1]=min(cvt);
a1=aa(a1);
vd=(va2+a1).^(-1);
vc1=repmat((va2.*vd),n,1);
Pa1=(psi.*vc1)*psi'/n;
u=(eye(n)-Pa1)*yy;
sigu2=u'*u/n;
dh=inv(yy'*Pa1*yy)*(yy'*Pa1*y);
sigeps2=(y-dh*yy)'*(y-dh*yy)/n;
sigueps=u'*(y-dh*yy)/n;
v=u-(y-dh*yy).*(sigueps/sigeps2);
sigv2=v'*v/n;

for a=1:1:nt;
st(a)=(sigueps^2)*(tra(a,1)^2)/n+sigeps2*(cvt(a)-sigu2*tra(a,2)/n);
end;
for b=1:1:nt;
stl(b)= sigeps2*(sigv2*tra(b,2)/n + cvt(b)- sigu2*tra(b,2)/n);
end;

[min1, alph]=min(st);
alpha=aa(alph);
aopt(it)=alpha;
%%%Regularized parameter for T LIML
[min1l, alphl]=min(stl);
alphal=aa(alphl);
aoptl(it)=alphal;

vo=(va2+alpha).^(-1);
vc2=repmat((va2.*vo),n,1);
Pao=(psi.*vc2)*psi'/n;    %Kn_alpha inverse
wh=Pao*yy;
d_t(it)=inv(wh'*yy)*wh'*y;
v_t(it)=((y-yy*d_t(it))'*(y-yy*d_t(it))/n)*((wh'*wh)/(yy'*wh)^2);


%%%Estimation of the T LIML
vol=(va2+alphal).^(-1);
qt=va2.*vol;
vc2l=repmat((va2.*vol),n,1);
Paol=(psi.*vc2l)*psi'/n;    %Kn_alpha inverse
whl=Paol*yy;

%% ZT: fminsearch was used
% LIML was used as the inital value
[rt(it), mut]=fminsearch(@(x)obj_fun(y,yy,Paol,x),d_il(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_tl(it)=(yy'*(Paol-mut*eye(n))*y)/(yy'*(Paol-mut*eye(n))*yy);

whlm=(Paol-mut*eye(n))*yy;
%d_tl(it)=rt(it);
v_tl(it)=((y-d_tl(it)*yy)'*(y-d_tl(it)*yy)/n)*((whlm'*whlm)/(yy'*whlm)^2);



%%% Landweber Fridman %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
knorm=max(va);
c=0.1/knorm^2;


for a=1:1:nlf;
coef=(1-(1-c*va2).^a);
vc=repmat((coef),n,1);
Pa=(psi.*vc)*psi'/n; 

u=(eye(n)-Pa)*yy;
cvlf(a)=(u'*u/n)/(1-sum(diag(Pa))/n)^2;
trlf(a,:)=[sum(diag(Pa)) sum(diag(Pa*Pa))];
end;
[min3, a1]=min(cvlf);
coef1=(1-(1-c*va2).^a1);
vc1=repmat((coef1),n,1);
Pa1=(psi.*vc1)*psi'/n;
u=(eye(n)-Pa1)*yy;
sigu2=u'*u/n;
dh=(yy'*Pa1*y)/(yy'*Pa1*yy);
sigeps2=(y-dh*yy)'*(y-dh*yy)/n;
sigueps=u'*(y-dh*yy)/n;
v=u-(y-dh*yy).*(sigueps/sigeps2);
sigv2=v'*v/n;

for a=1:1:nlf;
slf(a)=(sigueps^2)*(trlf(a,1)^2)/n+sigeps2*(cvlf(a)-sigu2*trlf(a,2)/n);
end;
for b=1:1:nlf;
slfl(b)= sigeps2*(sigv2*trlf(b,2)/n + cvlf(b)- sigu2*trlf(b,2)/n);
end;
[min4, lnn]=min(slf);
lopt(it)=lnn;
%%%selction of regularized parameter
[min4, lnnl]=min(slfl);
loptl(it)=lnnl;


coef=(1-(1-c*va2).^lnn);
vc=repmat((coef),n,1);
Pal=(psi.*vc)*psi'/n;

wh=Pal*yy;
d_l(it)=inv(wh'*yy)*(wh'*y);
v_l(it)=(y-yy*d_l(it))'*(y-yy*d_l(it))/n*(wh'*wh)/(yy'*wh)^2;

%%%Estimation regulirized LIML

% ZT: the second time RLIML
coef=(1-(1-c*va2).^lnnl);
vc=repmat((coef),n,1);
Pall=(psi.*vc)*psi'/n;
whl=Pall*yy;
[rl(it), mul]=fminsearch(@(h)obj_fun(y,yy,Pall,h),d_il(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_ll(it)=(yy'*(Pall-mul*eye(n))*y)/(yy'*(Pall-mul*eye(n))*yy);
whlm=(Pall-mul*eye(n))*yy;

v_ll(it)=((y-d_ll(it)*yy)'*(y-d_ll(it)*yy)/n)*((whlm'*whlm)/(yy'*whlm)^2);

%%Principal components%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

psi1=psi;

% preliminary estimator of sigma_u and sigma_ueps 

for k=1:1:kpc;
fi=psi1(:,1:k); % the psi are orthonormal
Pa1=fi*(fi'*fi)^(-1)*fi';

u=(eye(n)-Pa1)*yy;
cvpc(k)=(u'*u/n)/(1-sum(diag(Pa1))/n)^2;
trpc(k,:)=[sum(diag(Pa1)) sum(diag(Pa1*Pa1))];
end;

[min1, k1]=min(cvpc);
fi1=psi1(:,1:k1);
%P1=fi1*fi1'/n;
P1=fi1*(fi1'*fi1)^(-1)*fi1';
u=(eye(n)-P1)*yy;
sigu2=u'*u/n;
dh=inv(yy'*P1*yy)*(yy'*P1*y);
sigeps2=(y-dh*yy)'*(y-dh*yy)/n;
sigueps=u'*(y-dh*yy)/n;
v=u-(y-dh*yy).*(sigueps/sigeps2);
sigv2=v'*v/n;

for a=1:1:kpc;
spc(a)=(sigueps^2)*(a^2)/n + sigeps2*(cvpc(a)-sigu2*a/n);

end;

[min1, ko]=min(spc);
kopt(it)=ko;
fo=psi1(:,1:ko);
%Po=fo*fo'/n; 
Po=fo*(fo'*fo)^(-1)*fo';
wh=Po*yy;
d_pc(it)=(wh'*y)/(wh'*yy);
v_pc(it)=((y-yy*d_pc(it))'*(y-yy*d_pc(it))/n)/(wh'*yy);


for c=1:1:kpc;
 spcl(c)= sigeps2*(sigv2*trpc(c,2)/n + cvpc(c)- sigu2*trpc(c,2)/n);
end
[minp, kol]=min(spcl);
koptl(it)=kol;

%%%Estimation LIML PC
fol=psi1(:,1:kol);
%Pol=fol*fol'/n;
Pol=fol*(fol'*fol)^(-1)*fol';
whl=Pol*yy;
[rpc(it), mup]=fminsearch(@(x)obj_fun(y,yy,Pol,x),d_il(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_pcl(it)=(yy'*(Pol-mup*eye(n))*y)/(yy'*(Pol-mup*eye(n))*yy);
%d_pcl(it)=rpc(it);
whlm=(Pol-mup*eye(n))*yy;
v_pcl(it)=((y-d_pcl(it)*yy)'*(y-d_pcl(it)*yy)/n)/(whlm'*yy);


%%% Donald and Newey estimator %%%%%
%%% use cross validation criterion to compute R(K) on the first stage %%

for k=1:1:kbar;
    fi=x(:,1:k);
    
    pk=fi*inv(fi'*fi)*fi';
    ut=(eye(n)-pk)*yy;
    cvsls(k)=((ut'*ut)/n)/(1-k/n)^2;
end;
[minq, ksls]=min(cvsls);
kv(it)=ksls;
fik=x(:,1:ksls);
pkopt=fik*inv(fik'*fik)*fik';

ut2=(eye(n)-pkopt)*yy;
d2=inv(yy'*pkopt*yy)*yy'*pkopt*y;
e2=y-yy*d2;
sigeps2=e2'*e2/n; sigu2=ut2'*ut2/n; sigueps=ut2'*e2/n; 
 vt=ut-(y-dh*yy).*(sigueps/sigeps2);
sigv2=vt'*vt/n;                     %Page 1164

% MSE for DN estimator
for k=1:1:kbar;
    ssls(k)=(sigueps^2)*(k^2)/n + sigeps2*(cvsls(k)-sigu2*k/n);
end;
for c=1:1:kpc;
 liml(c)= sigeps2*(-sigueps^2*c/(n*sigeps2) + cvsls(c));
end
[mins, km]=min(ssls);
kv(it)=km;
%%%Estimation of regularized param
[mins, kl]=min(liml);
kli(it)=kl;

fik1=x(:,1:km);

pkopt=fik1*inv(fik1'*fik1)*fik1';

% Donald and Newey estimator 

yh=pkopt'*yy;

d_sls(it)=inv(yh'*yy)*yh'*y; %2SLS estimator
v_sls(it)=(1/n)*((y-yy*d_sls(it))'*(y-yy*d_sls(it)))/(yy'*yh);  %variance

%%% Estimation of R LIML DN
fikliml=x(:,1:kl);
pkopt1=fikliml*inv(fikliml'*fikliml)*fikliml';
yhl=pkopt1'*yy;
[rp(it), mun]=fminsearch(@(x)obj_fun(y,yy,pkopt1,x),d_il(it),optimset('MaxIter',5000,'MaxFunEvals',5000));
d_liml(it)=(yy'*(pkopt1-mun*eye(n))*y)/(yy'*(pkopt1-mun*eye(n))*yy);
%d_liml(it)=rp(it);
yhlm=(pkopt1-mun*eye(n))*yy;
v_liml(it)=((y-d_liml(it)*yy)'*(y-d_liml(it)*yy)/n)/(yy'*yhlm);

end


%%%Summary of results.

gamh=[d_t d_l d_pc d_sls d_iv]; vgam=[v_t v_l v_pc v_sls v_iv]; vgamB=[V_Bt V_Bl V_Bpc V_Bsls V_Biv];
% median bias 
med=median(gamh)-gam;
%median absolute bias  
mab=median(abs(gamh-gam));
% dispersion range 
dis=quantile(gamh,0.9)-quantile(gamh,0.1);
% coverage rate 
for i=1:1:itn
 for j=1:1:5   
 if gamh(i,j)> 0.1-1.96*sqrt(vgam(i,j))&& gamh(i,j) < 0.1+1.96*sqrt(vgam(i,j));
    C(i,j)=1;
 else
     C(i,j)=0;

 end
 end 
end
cov=mean(C);
for i=1:1:itn
 for j=1:1:5   
 if gamh(i,j)> 0.1-1.96*sqrt(vgamB(i,j))&& gamh(i,j) < 0.1+1.96*sqrt(vgamB(i,j));
    CB(i,j)=1;
 else
     CB(i,j)=0;

 end
 end 
end
covb=mean(CB);
%if gamh(i)> 0.1-1.96*sqrt(vgam)||gamh(j) < 0.1+1.96*sqrt(vgam);
%cov=mean(C);
%MSE 
A=(gamh-gam).*(gamh-gam);
mse= mean(A);
%%%Summary
delh=[d_tl d_ll d_pcl d_liml d_Liml]; vdel=[v_tl v_ll v_pcl v_liml v_BLiml]; vgamBl=[V_Btl V_Bll V_Bpcl V_Bliml V_BLiml];
% median bias 
medl=median(delh)-gam;
%median absolute bias  
mabl=median(abs(delh-gam));
% dispersion range 
disl=quantile(delh,0.9)-quantile(delh,0.1);
% coverage rate
for i=1:1:itn
 for j=1:1:5   
 if delh(i,j)> 0.1-1.96*sqrt(vdel(i,j))&& delh(i,j) < 0.1+1.96*sqrt(vdel(i,j));
    Cl(i,j)=1;
 else
     Cl(i,j)=0;

 end
 
 end 
end
covl=mean(Cl);
for i=1:1:itn
 for j=1:1:5   
 if delh(i,j)> 0.1-1.96*sqrt(vgamBl(i,j))&& delh(i,j) < 0.1+1.96*sqrt(vgamBl(i,j));
    CBl(i,j)=1;
 else
     CBl(i,j)=0;

 end
 
 end 
end
covbl=mean(CBl);
%MSE 
Al=(delh-gam).*(delh-gam);
msel= mean(Al);

disp('---------------------------------------------------------------------')
disp( ' size(n)')
disp(n) 
disp( 'Number of simulations model 1b')
disp(itn)
disp('---------------------------------------------------------------------')
disp( 'Number of instruments model 1b')
disp(kbar)
disp('---------------------------------------------------------------------')
disp( 'medb  medabdev   disp_range  MSE  Cov CovB')

disp([med' mab' dis' mse' cov' covb']);
%disp([med' mab' dis' cov']);
disp('---------------------------------------------------------------------')
%disp( 'median_biasl  median_absolute_dev   dispersion_range  MSE')

disp([medl' mabl' disl' msel' covl' covbl']);
%disp([medl' mabl' disl' covl']);
%%%% Summary result Regularized parameter%%%
alpha=[aopt lopt kopt kv];
alphal=[aoptl' loptl' koptl' kli'];
m=mean(alpha);
ml=mean(alphal);
s=std(alpha);
sl=std(alphal);
q1=quantile(alpha,0.25);
q2=median(alpha);
q3=quantile(alpha,0.75);
q1l=quantile(alphal,0.25);
q2l=median(alphal);
q3l=quantile(alphal,0.75);
mod=mode(alpha);
modl=mode(alphal);
disp('---------------------------------------------------------------------')
disp( ' size(n) Properties 2sls')
disp( 'mean  Sd   q1  q2 q3 mode')
disp([m' s' q1' q2' q3' mod']);
disp('---------------------------------------------------------------------')
disp('---------------------------------------------------------------------')
disp( ' size(n) Properties LIML')
disp([ml' sl' q1l' q2l' q3l' modl']);
disp('---------------------------------------------------------------------')



TSLS=[med' mab' dis' mse' cov' covb'];
LIML=[medl' mabl' disl' msel' covl' covbl'];

TSLS1=[m' s' q1' q2' q3'];
LIML1=[ml' sl' q1l' q2l' q3l'];

MR(:,:,t)=[TSLS' LIML'];

MRp(:,:,t)=[TSLS1' LIML1'];


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Genarated tex table%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%Table for estimators performances%%%%%
VD(1,:)=char('T2SL');
VD(2,1:length(char('L2LS')))=char('L2LS');
VD(3,1:length(char('P2LS')))=char('P2LS');
VD(4,1:length(char('D2LS')))=char('D2LS');
VD(5,1:length(char('IV')))=char('IV');
VD(6,1:length('TLIML'))=char('TLIML');
VD(7,1:length(char('LLIML')))=char('LLIML');
VD(8,1:length(char('PLIML')))=char('PLIML');
VD(9,1:length(char('DLIML')))=char('DLIML');
VD(10,1:length(char('LIML')))=char('LIML');

CA(1,:)=char('Med.bias');
CA(2,1:length(char('Med.abs')))=char('Med.abs');
CA(3,1:length(char('Disp')))=char('Disp');
CA(4,1:length(char('MSE')))=char('MSE');
CA(5,1:length(char('Cov')))=char('Cov');
CA(6,1:length(char('Cov_B')))=char('Cov_B');

PT(1,:)=char('Mean');
PT(2,1:length(char('sd')))=char('sd');
PT(3,1:length(char('q1')))=char('q1');
PT(4,1:length(char('q2')))=char('q2');
PT(5,1:length(char('q3')))=char('q3');

filename = ['C:\Users\gt223\Documents\Recherche\Theory\LIML\paper Guy_Marine\Code_RLIML\TableLIML1b.TeX'];
fidTeX = fopen(filename,'w');
%fprintf(fidTeX,'%% TeX-table generated by dynare_estimation (Dynare).\n');
fprintf(fidTeX,'%% RESULTS LIML \n');
fprintf(fidTeX,['%% ' datestr(now,0)]);
fprintf(fidTeX,' \n');
fprintf(fidTeX,' \n');

fprintf(fidTeX,'\\begin{table}\n');
fprintf(fidTeX,'\\centering\n');
fprintf(fidTeX,'\\caption{Simulations results of Model 1 b   with $R^2_f=0.1$, $n=500$, 1000 replications}\n ');
%fprintf(fidTeX,'{\\scriptsize \n');
fprintf(fidTeX,'{\\tiny \n');
fprintf(fidTeX,'\\begin{tabular}{rcrrrrrrrrrr} \n');
    fprintf(fidTeX,'\\addlinespace  \\toprule \n');
    fprintf(fidTeX,[ '%s & %s & %s & %s & %s  & %s & %s & %s & %s & %s & %s  & %s \\\\ \n'],...
        char('Model 1b'),...        
        char(' '),...
        deblank(VD(1,:)),...
        deblank(VD(2,:)),...
        deblank(VD(3,:)),...
        deblank(VD(4,:)),...
        deblank(VD(5,:)),...
        deblank(VD(6,:)),...
        deblank(VD(7,:)),...
        deblank(VD(8,:)),...
        deblank(VD(9,:)),...
        deblank(VD(10,:)));
        
 %==========================================================================%
    fprintf(fidTeX,'\\midrule  \n');
    %fprintf(fidTeX,'\\label{Table:VarianceDecomposition:%s}\n',int2str(Steps(i)));
    
    for il=1:length(L)
    fprintf(fidTeX,['\\multirow{6}[2]{*}{L=' int2str(L(il)) '}\n']);
    %fprintf(fidTeX,'\\multirow{5}[2]{*}{L=15}');
    %fprintf(fidTeX,['\\multirow{5}[2]{*}{L=' L(il) '}\n']);
    

        for ip=1:6
            fprintf(fidTeX,[ '& %s & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f \\\\ \n'],...
                deblank(CA(ip,:)),...
                MR(ip,1,il),...
                MR(ip,2,il),...
                MR(ip,3,il),...
                MR(ip,4,il),...
                MR(ip,5,il),...
                MR(ip,6,il),...
                MR(ip,7,il),...
                MR(ip,8,il),...
                MR(ip,9,il),...
                MR(ip,10,il));
        end
        fprintf(fidTeX,'\\midrule  \n');
    end
    %==========================================================================%


%**************************************************************************%
fprintf(fidTeX,'\\bottomrule \n');
fprintf(fidTeX,'\\end{tabular}\n ');
fprintf(fidTeX,'} \n');
fprintf(fidTeX,['\\label{Table:model1}\n']);
fprintf(fidTeX,'\\end{table}\n');

fprintf(fidTeX,'%% End of TeX file.\n');
fclose(fidTeX);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Table of properties of alpha%%%%%%%%%%%%%%
filename = ['C:\Users\gt223\Documents\Recherche\Theory\LIML\paper Guy_Marine\Code_RLIML\TableLIML1bp.TeX'];
fidTeX = fopen(filename,'w');
%fprintf(fidTeX,'%% TeX-table generated by dynare_estimation (Dynare).\n');
fprintf(fidTeX,'%% RESULTS LIML \n');
fprintf(fidTeX,['%% ' datestr(now,0)]);
fprintf(fidTeX,' \n');
fprintf(fidTeX,' \n');

fprintf(fidTeX,'\\begin{table}\n');
fprintf(fidTeX,'\\centering\n');
fprintf(fidTeX,'\\caption{Properties of the distribution of the regularization parameters Model 1b}\n ');
%fprintf(fidTeX,'{\\scriptsize \n');
fprintf(fidTeX,'{\\tiny \n');
fprintf(fidTeX,'\\begin{tabular}{rcrrrrrrrr} \n');
    fprintf(fidTeX,'\\addlinespace  \\toprule \n');
    fprintf(fidTeX,[ '%s & %s & %s & %s & %s  & %s & %s & %s & %s & %s  \\\\ \n'],...
        char('Model 1b'),...        
        char(' '),...
        deblank(VD(1,:)),...
        deblank(VD(2,:)),...
        deblank(VD(3,:)),...
        deblank(VD(4,:)),...
        deblank(VD(6,:)),...
        deblank(VD(7,:)),...
        deblank(VD(8,:)),...
        deblank(VD(9,:)));
        
 %==========================================================================%
    fprintf(fidTeX,'\\midrule  \n');
    %fprintf(fidTeX,'\\label{Table:VarianceDecomposition:%s}\n',int2str(Steps(i)));
    
    for ik=1:length(L)
    fprintf(fidTeX,['\\multirow{5}[2]{*}{L=' int2str(L(ik)) '}\n']);
    %fprintf(fidTeX,'\\multirow{5}[2]{*}{L=15}');
    %fprintf(fidTeX,['\\multirow{5}[2]{*}{L=' L(il) '}\n']);
    

        for ir=1:5
            fprintf(fidTeX,('& %s & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f & %6.3f  \\\\ \n'),...
                deblank(PT(ir,:)),...
                MRp(ir,1,ik),...
                MRp(ir,2,ik),...
                MRp(ir,3,ik),...
                MRp(ir,4,ik),...
                MRp(ir,5,ik),...
                MRp(ir,6,ik),...
                MRp(ir,7,ik),...
                MRp(ir,8,ik));
        end
        fprintf(fidTeX,'\\midrule  \n');
    end
    %==========================================================================%


%**************************************************************************%
fprintf(fidTeX,'\\bottomrule \n');
fprintf(fidTeX,'\\end{tabular}\n ');
fprintf(fidTeX,'} \n');
fprintf(fidTeX,['\\label{Table:model1}\n']);
fprintf(fidTeX,'\\end{table}\n');

fprintf(fidTeX,'%% End of TeX file.\n');
fclose(fidTeX);

toc