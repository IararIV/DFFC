function  xNew=condTVmean(projections,meanFF,FF,DF,x,DS)
% condTVmean finds the optimal estimates  of the coefficients of the 
% eigen flat fields.

%downsample images
projections=imresize(projections, 1/DS);
meanFF=imresize(meanFF, 1/DS);
FF2=zeros(size(meanFF,1),size(meanFF,2),size(FF,3));
for ii=1:size(FF,3)
    FF2(:,:,ii)=imresize(FF(:,:,ii), 1/DS);
end
FF=FF2;
DF=imresize(DF, 1/DS);

%optimize coefficients
func=@(X)fun(projections,meanFF,FF,DF,X);
xNew=fminunc(func,x);
end


function cost=fun(projections,meanFF,FF,DF,x)
%objective function
FF_eff=zeros(size(FF,1),size(FF,2));
for  ii=1:size(FF,3)
    FF_eff=FF_eff+x(ii)*FF(:,:,ii);
end
logCorProj=(projections-DF)./(meanFF+FF_eff)*mean(meanFF(:)+FF_eff(:));
[Gx,Gy]=gradient(logCorProj);
mag=sqrt(Gx.^2+Gy.^2); 
cost=sum(mag(:));
end
