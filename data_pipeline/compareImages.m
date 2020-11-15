function s = compareImages(I, thresh)
    J = double(reshape(I, size(I,1), []));
    s = abs(J * J' - sum(J .* J,2)) <= thresh;
end