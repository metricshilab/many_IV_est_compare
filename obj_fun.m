function f=obj_fun(y,yy,P_a,x)
d=x;
f=(y-yy*d)'*(P_a)*(y-yy*d)*((y-yy*d)'*(y-yy*d))^(-1);