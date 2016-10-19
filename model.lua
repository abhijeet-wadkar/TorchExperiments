-- neural network for classifying the 
-- amazon aws dataset for torch
-- there are images of airplane, automobile, bird, cat, deer, dog,
-- frog, horse, ship, truck. Total 10 classes

require 'nn'
require 'image'

-- step 1: load the data
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- TODO: later needs to understand meaning of setmetatable()
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
testset.data = testset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

print('Some characteristics of data')
print(trainset)
print(trainset:size())

-- normalize the data if needed(optional but important step)
-- will do it afterwards

-- step 2: define network
model = nn.Sequential()

--[===[
model:add(nn.Reshape(3*32*32))
model:add(nn.Linear(3*32*32, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())
--]===]

model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 6, 5, 5, 1, 1, 2, 2)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialConvolution(6, 16, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(32*4*4))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model:add(nn.Linear(32*4*4, 120))             -- fully connected layer (matrix multiplication between input and weights)
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

print(model)

-- step 3: define loss function
criterion = nn.ClassNLLCriterion()

-- step 4: train network on training data
--[====[
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 15
trainer:train(trainset)
--]====]


-- train manually
-- usally we loop over the data and update the parameters of the network

model:training()
model:updateParameters(0.001)

imageNo = 1
numTraining = trainset:size()
print("numTraining: "..numTraining)
batchSize = 100
numBatches = math.floor(numTraining/batchSize)
numberOfIterations = 10

for index = 1, numberOfIterations*numBatches do

	-- load the batch of data
	if imageNo >= numTraining then
        imageNo = 1
    end

    -- get the pointer to the batch in origial tensor
    imgs = trainset.data:narrow(1, imageNo, batchSize)
    labels = trainset.label:narrow(1, imageNo, batchSize)

    -- train for current batch
	model:zeroGradParameters()
	outputs = model:forward(imgs)
	loss = criterion:forward(outputs, labels)
	criterion_gradient = criterion:backward(outputs, labels)
    model:backward(imgs, criterion_gradient)

	print("Current loss: "..loss)

    imageNo = imageNo + batchSize
end

-- step 5: test network with test data
correct = 0
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = model:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
        correct = correct + 1
    end
end
print('Accuracy: ')
print(correct, 100*correct/10000 .. ' % ')


for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end