function hybrid_elpv_classification(dataset_path)
    % dataset_path: path to folder containing 'functional' and 'defective' subfolders
    % Example: 'D:\00 - MATLAB\elpv-dataset-master\elpv-dataset-master\src\elpv_dataset\data_organized'

    % 1. Create a file datastore to list all images with labels
    imds = imageDatastore(fullfile(dataset_path, {'functional','defective'}), ...
        'LabelSource', 'foldernames', 'IncludeSubfolders', true);

    % Get all files and labels
    allFiles = imds.Files;
    allLabels = imds.Labels;

    % 2. Split into train/test (80/20) using the same indices
    rng(42);
    cvp = cvpartition(allLabels, 'Holdout', 0.2);
    trainIdx = training(cvp);
    testIdx = test(cvp);

    trainFiles = allFiles(trainIdx);
    testFiles = allFiles(testIdx);
    trainLabels = allLabels(trainIdx);
    testLabels = allLabels(testIdx);

    % 3. Create separate datastores for morphology (grayscale) and deep (RGB)
    % Morphology datastore: reads grayscale double
    imdsTrainMorph = imageDatastore(trainFiles, 'ReadFcn', @(f) im2double(imread(f)));
    imdsTestMorph  = imageDatastore(testFiles,  'ReadFcn', @(f) im2double(imread(f)));

    % Deep datastore: reads RGB (by replicating grayscale)
    imdsTrainDeep = imageDatastore(trainFiles, 'ReadFcn', @(f) repmat(im2double(imread(f)), [1 1 3]));
    imdsTestDeep  = imageDatastore(testFiles,  'ReadFcn', @(f) repmat(im2double(imread(f)), [1 1 3]));

    % 4. Load pretrained network (SqueezeNet if available, else ResNet-50)
    try
        net = squeezenet;
        featureLayer = 'pool10';
        inputSize = net.Layers(1).InputSize;
        fprintf('Using SqueezeNet for deep features.\n');
    catch
        net = resnet50;
        featureLayer = 'fc1000';
        inputSize = net.Layers(1).InputSize;
        fprintf('Using ResNet-50 for deep features.\n');
    end

    % 5. Create augmented datastores for deep feature extraction (resize only)
    augimdsTrainDeep = augmentedImageDatastore(inputSize(1:2), imdsTrainDeep);
    augimdsTestDeep  = augmentedImageDatastore(inputSize(1:2), imdsTestDeep);

    % 6. Extract deep features
    fprintf('Extracting deep features...\n');
    trainDeep = activations(net, augimdsTrainDeep, featureLayer, 'OutputAs', 'rows');
    testDeep  = activations(net, augimdsTestDeep,  featureLayer, 'OutputAs', 'rows');

    % 7. Extract morphological features
    fprintf('Extracting morphological features...\n');
    trainMorph = [];
    testMorph = [];
    for i = 1:numel(imdsTrainMorph.Files)
        img = readimage(imdsTrainMorph, i);
        trainMorph = [trainMorph; extract_morph_features(img)];
    end
    for i = 1:numel(imdsTestMorph.Files)
        img = readimage(imdsTestMorph, i);
        testMorph = [testMorph; extract_morph_features(img)];
    end

    % 8. Manual normalization of each feature set
    % Deep features
    mu_d = mean(trainDeep, 1);
    sigma_d = std(trainDeep, 0, 1);
    sigma_d(sigma_d == 0) = 1;
    trainDeep = (trainDeep - mu_d) ./ sigma_d;
    testDeep  = (testDeep - mu_d) ./ sigma_d;

    % Morphological features
    mu_m = mean(trainMorph, 1);
    sigma_m = std(trainMorph, 0, 1);
    sigma_m(sigma_m == 0) = 1;
    trainMorph = (trainMorph - mu_m) ./ sigma_m;
    testMorph  = (testMorph - mu_m) ./ sigma_m;

    % 9. Concatenate features
    trainFeatures = [trainDeep, trainMorph];
    testFeatures  = [testDeep, testMorph];

    % 10. Train a simple neural network on the combined features
    layers = [
        featureInputLayer(size(trainFeatures,2))
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.001, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    fprintf('Training hybrid classifier...\n');
    nnModel = trainNetwork(trainFeatures, trainLabels, layers, options);

    % 11. Evaluate
    predLabels = classify(nnModel, testFeatures);
    accuracy = sum(predLabels == testLabels) / numel(testLabels);
    fprintf('\nHybrid (deep + morph) accuracy: %.2f%%\n', accuracy*100);

    % 12. Confusion matrix
    figure;
    confusionchart(testLabels, predLabels);
    title('Hybrid: Deep + Morphological Features');
end