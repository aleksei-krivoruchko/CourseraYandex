#nowarn "211"
#I __SOURCE_DIRECTORY__
#I "../Deedle/lib/net40"
#I "../Deedle.1.2.5/lib/net40"
#I "../Deedle.RPlugin/lib/net40"
#I "../Deedle.RPlugin.1.2.5/lib/net40"
#I "../FSharp.Charting/lib/net40"
#I "../FSharp.Charting.0.90.14/lib/net40"
#I "../FSharp.Data/lib/net40"
#I "../FSharp.Data.2.3.1/lib/net40"
#I "../MathNet.Numerics/lib/net40"
#I "../MathNet.Numerics.3.12.0/lib/net40"
#I "../MathNet.Numerics.FSharp/lib/net40"
#I "../MathNet.Numerics.FSharp.3.12.0/lib/net40"
#I "../DynamicInterop/lib/net40"
#I "../DynamicInterop.0.7.4/lib/net40"
#I "../R.NET.Community/lib/net40"
#I "../R.NET.Community.1.6.5/lib/net40"
#I "../R.NET.Community.FSharp/lib/net40"
#I "../R.NET.Community.FSharp.1.6.5/lib/net40"
#I "../RProvider/lib/net40"
#I "../RProvider.1.1.20/lib/net40"
#I "../Suave/lib/net40"
#I "../Suave.1.1.3/lib/net40"
#I "../XPlot.Plotly/lib/net45"
#I "../XPlot.Plotly.1.3.1/lib/net45"
#I "../XPlot.GoogleCharts/lib/net45"
#I "../XPlot.GoogleCharts.1.3.1/lib/net45"
#I "../XPlot.GoogleCharts.Deedle/lib/net45"
#I "../XPlot.GoogleCharts.Deedle.1.3.1/lib/net45"
#I "../Google.DataTable.Net.Wrapper/lib"
#I "../Google.DataTable.Net.Wrapper.3.1.2.0/lib"
#I "../Newtonsoft.Json/lib/net40"
#I "../Newtonsoft.Json.9.0.1/lib/net40"
#r "Deedle.dll"
#r "Deedle.RProvider.Plugin.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#r "FSharp.Charting.dll"
#r "FSharp.Data.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"
#r "DynamicInterop.dll"
#r "RDotNet.dll"
#r "RDotNet.NativeLibrary.dll"
#r "RDotNet.FSharp.dll"
#r "RProvider.Runtime.dll"
#r "RProvider.dll"
#r "Suave.dll"
#r "XPlot.Plotly.dll"
#r "XPlot.GoogleCharts.dll"
#r "XPlot.GoogleCharts.Deedle.dll"
#r "Google.DataTable.Net.Wrapper.dll"
#r "Newtonsoft.Json.dll"

#load "Shared/Server.fsx"
#load "Shared/Styles.fsx"
#if NO_FSI_ADDPRINTER
#else
#if HAS_FSI_ADDHTMLPRINTER
#load "Html/Charting.fsx"
#load "Html/Deedle.fsx"
#load "Html/MathNet.fsx"
#load "Html/Text.fsx"
#load "Html/XPlot.fsx"
#else
#load "Text/FsLab.fsx"
#endif
#endif

namespace FSharp.Charting
open FSharp.Charting
open Deedle

[<AutoOpen>]
module FsLabExtensions =
  type FSharp.Charting.Chart with
    static member Line(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Line(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Column(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Column(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Pie(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Pie(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Area(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Area(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Bar(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Bar(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)

namespace MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra
open Deedle

module Matrix =
  let inline toFrame matrix = matrix |> Matrix.toArray2 |> Frame.ofArray2D
module DenseMatrix =
  let inline ofFrame frame = frame |> Frame.toArray2D |> DenseMatrix.ofArray2
module SparseMatrix =
  let inline ofFrame frame = frame |> Frame.toArray2D |> SparseMatrix.ofArray2
module Vector =
  let inline toSeries vector = vector |> Vector.toSeq |> Series.ofValues
module DenseVector =
  let inline ofSeries series = series |> Series.values |> Seq.map (float) |> DenseVector.ofSeq
module SparseVector =
  let inline ofSeries series = series |> Series.values |> Seq.map (float) |> SparseVector.ofSeq

namespace Deedle
open Deedle
open MathNet.Numerics.LinearAlgebra

module Frame =
  let inline ofMatrix matrix = matrix |> Matrix.toArray2 |> Frame.ofArray2D
  let inline toMatrix frame = frame |> Frame.toArray2D |> DenseMatrix.ofArray2

  let ofCsvRows (data:FSharp.Data.Runtime.CsvFile<'T>) =
    match data.Headers with
    | None -> Frame.ofRecords data.Rows
    | Some names -> Frame.ofRecords data.Rows |> Frame.indexColsWith names

module Series =
  let inline ofVector vector = vector |> Vector.toSeq |> Series.ofValues
  let inline toVector series = series |> Series.values |> Seq.map (float) |> DenseVector.ofSeq
