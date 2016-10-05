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
print(#trainset.data)
print(trainset:size())

-- normalize the data if needed(optional but important step)
-- will do it afterwards

-- step 2: define network
model = nn.Sequential()
model:add(nn.Reshape(3*32*32))
model:add(nn.Linear(3*32*32, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())

print(model)

-- step 3: define loss function
criterion = nn.ClassNLLCriterion()


-- step 4: train network on training data
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.0001
trainer.maxIteration = 100
trainer:train(trainset)

-- step 5: test network with test data
print(classes[testset.label[100]])
predicted = model:forward(testset.data[100])
predicted:exp()

for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end