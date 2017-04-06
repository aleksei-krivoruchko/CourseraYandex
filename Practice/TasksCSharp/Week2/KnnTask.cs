using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;

namespace TasksCSharp.Week2
{
    public class KnnTask
    {
        public void Run()
        {
            var path = @"Data\wine.csv";
            var lines = File.ReadAllLines(path)
                .Skip(1)
                .Select(s=>s.Split(new [] {","}, StringSplitOptions.None).Select(double.Parse).ToArray())
                .ToArray();

            var inputs = lines.Select(a => a.Skip(1).ToArray()).ToArray();
            var outputs = lines.Select(a => (int)a[0]).ToArray();
            
            var results = new Dictionary<int, CrossValidationResult<object>>();

            for (int i = 1; i < 50; i++)
            {
                var kInd = i;
                var crossValidation = new CrossValidation(size: outputs.Length);

                crossValidation.Fitting = (k, indicesTrain, indicesValidation) =>
                {
                    var trainingInputs = inputs.Get(indicesTrain);
                    var trainingOutputs = outputs.Get(indicesTrain);

                    var validationInputs = inputs.Get(indicesValidation);
                    var validationOutputs = outputs.Get(indicesValidation);

                    var svm = new KernelSupportVectorMachine(new Polynomial(2), 2);

                    // Create a training algorithm and learn the training data
                    var knn = new KNearestNeighbors(k: kInd, classes: 3, inputs: trainingInputs, outputs: trainingOutputs);

                    var valErrors = 0;

                    for (int j = 0; j < validationInputs.Length - 1; j++)
                    {
                        var valIn = validationInputs[j];
                        var valComputed = knn.Compute(valIn);
                        var valExpected = validationOutputs[j];

                        if (valComputed != valExpected)
                        {
                            valErrors++;
                        }
                    }

                    var valError = valErrors / trainingInputs.Length;

                    return new CrossValidationValues(knn, 0, valError);
                };

                var result = crossValidation.Compute();
                results.Add(i, result);
                var trainingErrors = result.Training.Mean;
                var validationErrors = result.Validation.Mean;
            }

            var r = results.Min(p => p.Value.Validation.Mean);

        }
    }
}
