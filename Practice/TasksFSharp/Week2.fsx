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

let [<Literal>] CsvPath = __SOURCE_DIRECTORY__ + "\\titanic.csv"
let titanicItems = CsvProvider<CsvPath>.GetSample()

//

type DataItem(pClass:int, fare:decimal, age:float, sex:bool, survived:bool) = 
    member this.PClass = pClass
    member this.Fare = fare
    member this.Age = age
    member this.Sex = sex
    member this.Survived = survived
    
let items = 
  titanicItems.Rows 
  |>Seq.filter(fun i -> not <| System.Double.IsNaN i.Age)
  |>Seq.map(fun i -> new DataItem(i.Pclass, i.Fare, i.Age, i.Sex.Equals("male"), i.Survived))
  |>Seq.toArray
  

 //Обучите решающее дерево с параметром random_state=241 и 
 //остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
 
let data = new DataTable("Titanic data")


let features = 28 * 28
let classes = 10
 
let algorithm =
fun (svm: KernelSupportVectorMachine)
(classInputs: float[][])
(classOutputs: int[])
(i: int) (j: int) -> 
let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
strategy :> ISupportVectorMachineLearning
 
let kernel = Linear()
let svm = new MulticlassSupportVectorMachine(features, kernel, classes)
let learner = MulticlassSupportVectorLearning(svm, observations, labels)
let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
learner.Algorithm <- config
 
let error = learner.Run()
 
printfn "Error: %f" error


