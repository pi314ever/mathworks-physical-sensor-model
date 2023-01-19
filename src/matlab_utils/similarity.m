ref=imread("beach.png");
A = imnoise(ref,'salt & pepper', .02);
[peaksnr, snr] = psnr(A, ref);
[ssimval,ssimmap] = ssim(A,ref);
fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The Structural Similarity Index is %0.4f', ssimval);
montage({ref, A})

function [peaksnr, ssimval] = similarity(ref, A)
[peaksnr, snr] = psnr(A, ref);
[ssimval,ssimmap] = ssim(A,ref);
end