#load "../packages/FsLab.1.0.1/FsLab.fsx"
open Microsoft.FSharp.Linq
open FSharp.Data
open System
open System.Data

#r @"..\packages\Accord.3.4.0\lib\net46\Accord.dll"
#r @"..\packages\Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#r @"..\packages\Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#r @"..\packages\Accord.MachineLearning.3.4.0\lib\net46\Accord.MachineLearning.dll"

open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning
open Accord.MachineLearning.DecisionTrees
open Accord.MachineLearning.DecisionTrees.Learning
open Accord.Math
open Accord.Statistics.Analysis
open Accord.Statistics.Kernels
open Accord.Math.Optimization.Losses

let [<Literal>] CsvPath = __SOURCE_DIRECTORY__ + "\\titanic.csv"
let titanicItems = CsvProvider<CsvPath>.GetSample() 
  
let dataSet = titanicItems.Rows |> 
    Seq.filter(fun i -> not <| System.Double.IsNaN i.Age)
    |> Seq.map(fun i -> i, if i.Sex = "male" then 1.0 else 0.0 )

let inputs = dataSet |> 
    Seq.map(fun (i, sex) -> [| 
    double i.Pclass;
    double i.Fare;
    double i.Age;
    double sex
    |]) 
    |> Seq.toArray

let outputs = dataSet |> 
    Seq.map((fun (i, sex) -> if i.Survived then 1 else 0))
    |> Seq.toArray
  
//Обучите решающее дерево с параметром random_state=241 и 
//остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).

let teacher = new C45Learning();
//var teacher = new ID3Learning();

let tree = teacher.Learn(inputs, outputs);
let predicted = tree.Decide(inputs);
let x = new ZeroOneLoss(outputs);
let error =  x.Loss(tree.Decide(inputs));
let rules = tree.ToRules();
let ruleText = rules.ToString();
 
printfn "Error: %s" ruleText


