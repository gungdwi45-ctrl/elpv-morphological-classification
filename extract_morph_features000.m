function morph_features = extract_morph_features(img)
    % img: grayscale image (double, values in [0,1])
    % returns a row vector of morphological features

    % Parameters – adjust based on image resolution (300x300 here)
    disk_radii = [3, 5, 7, 10, 15];          % for blob detection
    line_lengths = [5, 10, 15, 20];           % for crack detection
    line_angles = [0, 45, 90, 135];           % orientations

    features = [];

    % ---- Blob features: top‑hat and bottom‑hat with disks ----
    for r = disk_radii
        se = strel('disk', r);
        % Top‑hat (bright blobs)
        I_top = imtophat(img, se);
        fea = compute_stats_and_morph(I_top);
        features = [features, fea];
        % Bottom‑hat (dark blobs)
        I_bot = imbothat(img, se);
        fea = compute_stats_and_morph(I_bot);
        features = [features, fea];
    end

    % ---- Line features: top‑hat with line SEs (cracks appear dark in EL?) ----
    % In EL images, cracks often appear as dark lines, so we might use bottom‑hat.
    % But to be safe, we compute both.
    for len = line_lengths
        for ang = line_angles
            se = strel('line', len, ang);
            % Top‑hat (bright lines – possibly not relevant but included for completeness)
            I_top = imtophat(img, se);
            fea = compute_stats_and_morph(I_top);
            features = [features, fea];
            % Bottom‑hat (dark lines – more likely for cracks)
            I_bot = imbothat(img, se);
            fea = compute_stats_and_morph(I_bot);
            features = [features, fea];
        end
    end

    morph_features = features;
end

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