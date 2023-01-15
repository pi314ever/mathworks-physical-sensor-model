img=imread("beach.png");
[x_dim, y_dim, z_dim]=size(img);
img_distorted=zeros(x_dim, y_dim, z_dim, class(img));

x_optic=round(x_dim/2);
y_optic=round(y_dim/2);
x_focal=round(x_dim/2);
y_focal=round(y_dim/2);

k1=0.01;
k2=0.01;
k3=-0.02;
r_max=sqrt(2);
scale=1+k1*(r_max^2)+k2*(r_max^4)+k3*(r_max^6);

for x = 1:x_dim
    for y = 1:y_dim
        % Normalize image coordinates
        x_norm=(x-x_optic)/x_focal;
        y_norm=(y-y_optic)/y_focal;
        r=sqrt(x_norm^2 + y_norm^2);

        % Perform distortion
        x_dist_norm=x_norm*(1+k1*(r^2) + k2*(r^4) + k3*(r^6))/scale;
        y_dist_norm=y_norm*(1+k1*(r^2) + k2*(r^4) + k3*(r^6))/scale;
        x_distorted=round(x_dist_norm*x_optic + x_optic);
        y_distorted=round(y_dist_norm*y_optic + y_optic);
        try
            img_distorted(x_distorted, y_distorted,:)=img(x,y,:);
        catch
        end
    end
end

% Now detect and remove black dots from image
img_distorted_double=im2double(img_distorted);
filled_img=img_distorted_double;
for iter=1:3
    filled_img(filled_img==0)=NaN;
    filled_img=fillmissing(filled_img,'movmedian', 3);
end