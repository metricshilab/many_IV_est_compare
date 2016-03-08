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




