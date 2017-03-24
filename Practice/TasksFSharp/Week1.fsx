#load "../packages/FsLab.1.0.1/FsLab.fsx"
open Microsoft.FSharp.Linq
open FSharp.Data
open System

let [<Literal>] CsvPath = __SOURCE_DIRECTORY__ + "\\titanic.csv"
let titanicItems = CsvProvider<CsvPath>.GetSample()

// 1

let manCount = 
  titanicItems.Rows 
  |> Seq.filter (fun elem -> elem.Sex.Equals("male"))
  |> Seq.length

let womanCount = 
  titanicItems.Rows 
  |> Seq.filter (fun elem -> elem.Sex.Equals("female"))
  |> Seq.length

// 2

let totalCount = titanicItems.Rows |> Seq.length |> double
let survivedCount = 
  titanicItems.Rows 
  |> Seq.filter (fun elem -> elem.Survived)
  |> Seq.length  |> double

let survivedPercent = (survivedCount / totalCount) * (100 |> double);

// 3

let firstClassCount = titanicItems.Rows |> Seq.filter (fun elem -> elem.Pclass = 1) |> Seq.length |> double
let firstClassPercent = (firstClassCount/totalCount)*(100 |> double);

// 4

let mean = 
  titanicItems.Rows 
  |> Seq.map (fun row -> row.Age) 
  |> Seq.filter (fun elem -> not (Double.IsNaN elem)) 
  |> Seq.average 





