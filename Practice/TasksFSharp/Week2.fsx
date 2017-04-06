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

let [<Literal>] CsvPath = __SOURCE_DIRECTORY__ + "\\wine.csv"
let items = CsvProvider<CsvPath>.GetSample() 

//let dataSet = titanicItems.Rows |> 
//    Seq.filter(fun i -> not <| System.Double.IsNaN i.Age)
//    |> Seq.map(fun i -> i, if i.Sex = "male" then 1.0 else 0.0 )





