using System;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using Accord;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.DecisionTrees.Rules;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Filters;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.TypeConversion;

namespace TasksCSharp.Week2
{
    public class DecisionTreeTask
    {
        public void Run()
        {
            var path = @"Data\titanic.csv";

            using (TextReader reader = File.OpenText(path))
            {
                try
                {
                    var csv = new CsvReader(reader);
                    csv.Configuration.Delimiter = ",";
                    csv.Configuration.Encoding = Encoding.UTF8;
                    csv.Configuration.HasHeaderRecord = true;
                    csv.Configuration.RegisterClassMap<TitanicCsvItemMap>();
                    var records = csv.GetRecords<TitanicCsvItem>().Where(i => i.Age != null).ToArray();
                
                    var inputs = records.Select(i => new [] {i.Pclass, i.Fare, i.Age.Value, i.IsMale ? 1.0 : 0.0}).ToArray();
                    var outputs = records.Select(i => i.Survived=="1"?1:0).ToArray();

                    var teacher = new C45Learning();
                    //var teacher = new ID3Learning();

                    var tree = teacher.Learn(inputs, outputs);
                    var predicted = tree.Decide(inputs);
                    var error = new ZeroOneLoss(outputs).Loss(tree.Decide(inputs));
                    var rules = tree.ToRules();
                    var ruleText = rules.ToString();

                    // submission: Sex Fare

                }
                catch (CsvTypeConverterException ex)
                {
                    foreach (var i in ex.Data)
                    {
                        var str = i.ToString();
                    }
                }
            }

        }
    }
}
