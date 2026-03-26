function fea = compute_stats_and_morph(img)
    % img: double image (output of morphological transform)
    % returns a vector of statistical and morphological features

    % Statistical features (5)
    stats = [mean(img(:)), std(img(:)), skewness(img(:)), ...
             kurtosis(img(:)), entropy(img(:))];

    % Morphological features from binary version
    bw = imbinarize(img);          % Otsu threshold
    bw = imopen(bw, strel('disk',1)); % remove tiny noise
    cc = bwconncomp(bw);
    numComp = cc.NumObjects;

    if numComp > 0
        props = regionprops(cc, 'Area', 'Eccentricity', 'Solidity', 'Extent');
        areas = [props.Area];
        ecc = [props.Eccentricity];
        sol = [props.Solidity];
        ext = [props.Extent];

        morph = [mean(areas), std(areas), mean(ecc), std(ecc), ...
                 mean(sol), std(sol), mean(ext), std(ext)];
    else
        morph = zeros(1,8);
    end

    % Add number of components as an extra feature
    morph = [numComp, morph];

    fea = [stats, morph];
end