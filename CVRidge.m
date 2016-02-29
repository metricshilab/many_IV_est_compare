function f = CVRidge(theta,y,x)

n = size(y,1);
ind = (1:n)';

f = 0;
for rr = 1:n
    yRTemp = y(ind ~= rr);
    xRTemp = x(ind ~= rr,:);
    bRidge = ridge(yRTemp,xRTemp,theta,0);
    f = f + (y(rr)-[1,x(rr,:)]*bRidge).^2;
end
