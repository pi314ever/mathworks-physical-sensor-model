img=imread("tomatoes.jpeg");
[x_dim, y_dim, zdim]=size(img);
img_distorted=zeros(x_dim, y_dim, z_dim, class(img));

x_optic=round(x_dim/2);
y_optic=round(y_dim/2);
x_focal=round(x_dim/2);
y_focal=round(y_dim/2);
p1=0.02;
p2=0.05;
r_max=sqrt(2);

x_scale=1+(2*p1+p2*(r_max^2+2));
y_scale=1+(p1*(r_max^2+2)+2*p2);

for x = 1:x_dim
    for y = 1:y_dim
        x_norm=(x-x_optic)/x_focal;
        y_norm=(y-y_optic)/y_focal;
        r=sqrt(x_norm^2 + y_norm^2);

        x_dist_norm=(x_norm+(2*p1*x_norm*y_norm+p2*(r^2+2*x_norm^2)))/x_scale;
        y_dist_norm=(y_norm+(p1*(r^2+2*y_norm^2)+2*p2*x_norm*y_norm))/y_scale;
        x_distorted=round(x_dist_norm*x_focal + x_optic);
        y_distorted=round(y_dist_norm*y_focal + y_optic);
        try
            img_distorted(x_distorted, y_distorted,:)=img(x,y,:);
        catch
            disp("dimension unmatched");
        end
    end
end
            
% Now detect and remove black dots from image
img_distorted_double=im2double(img_distorted);
img_distorted_double(img_distorted_double==0)=NaN;
filled_img=fillmissing(img_distorted_double,'movmedian', 3);
imshow(filled_img)
