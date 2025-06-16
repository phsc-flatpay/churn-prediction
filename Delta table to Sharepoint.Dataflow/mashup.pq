[StagingDefinition = [Kind = "FastCopy"]]
section Section1;
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Churn_Predictions_high_TPV_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "File", TransformationQueryName = "Churn_Predictions_high_TPV_DataDestinationTransform"]]]}]
shared Churn_Predictions_high_TPV = let
    // 1. Read the Delta table
    Source          = Lakehouse.Contents(null),
    #"Navigation 1" = Source{
                        [workspaceId = "e03dbe51-4f30-4e31-a84f-647f6b831f58"]
                     }[Data],
    #"Navigation 2" = #"Navigation 1"{
                        [lakehouseId = "54c8eb17-9ab5-4b14-9a01-728630a243fb"]
                     }[Data],
    #"Navigation 3" = #"Navigation 2"{
                        [Id = "Churn_Predictions_high_TPV", ItemKind = "Table"]
                     }[Data],

    // 2. Clean up
    #"Removed columns"  = Table.RemoveColumns(
                            #"Navigation 3",
                            {"__index_level_0__"}
                          ),
    #"Changed column type" = Table.TransformColumnTypes(
                               #"Removed columns",
                               {{"phonenumber", type text}}
                             ),

   // after you’ve already cast phonenumber to text
    #"Keep plus sign" =
        Table.TransformColumns(
            #"Changed column type",
            {{"phonenumber",
              each
                  // add a zero-width space (U+200B) right after the "+"
                  if Text.StartsWith(_, "+")
                  then "+" & Character.FromNumber(8203) & Text.Range(_, 1)
                  else _,
              type text}}
        )

in
    #"Keep plus sign";
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Churn_Predictions_low_TPV_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "File", TransformationQueryName = "Churn_Predictions_low_TPV_DataDestinationTransform"]]]}]
shared Churn_Predictions_low_TPV = let
    // 1. Read the Delta table
    Source          = Lakehouse.Contents(null),
    #"Navigation 1" = Source{
                        [workspaceId = "e03dbe51-4f30-4e31-a84f-647f6b831f58"]
                     }[Data],
    #"Navigation 2" = #"Navigation 1"{
                        [lakehouseId = "54c8eb17-9ab5-4b14-9a01-728630a243fb"]
                     }[Data],
    #"Navigation 3" = #"Navigation 2"{
                        [Id = "Churn_Predictions_low_TPV", ItemKind = "Table"]
                     }[Data],

    // 2. Clean up
    #"Removed columns"  = Table.RemoveColumns(
                            #"Navigation 3",
                            {"__index_level_0__"}
                          ),
    #"Changed column type" = Table.TransformColumnTypes(
                               #"Removed columns",
                               {{"phonenumber", type text}}
                             ),

   // after you’ve already cast phonenumber to text
    #"Keep plus sign" =
        Table.TransformColumns(
            #"Changed column type",
            {{"phonenumber",
              each
                  // add a zero-width space (U+200B) right after the "+"
                  if Text.StartsWith(_, "+")
                  then "+" & Character.FromNumber(8203) & Text.Range(_, 1)
                  else _,
              type text}}
        )

in
    #"Keep plus sign";
shared Churn_Predictions_high_TPV_DataDestination = let
  Pattern = SharePoint.Contents("https://flatpaydk.sharepoint.com/sites/Churn", [Implementation = "2.0"]),
  Navigation_1 = Pattern{[Name = "Shared Documents"]}[Content],
  FileNavigation = Navigation_1{[Name = "Churn_Predictions_high_TPV.csv"]}?[Content]?
in
  FileNavigation;
shared Churn_Predictions_high_TPV_DataDestinationTransform = (binaryStream, columnNameAndTypePairs) => let
  Pattern = Csv.Document(binaryStream, [Columns = List.Transform(columnNameAndTypePairs, each _{0}), CsvStyle = CsvStyle.QuoteAlways, IncludeByteOrderMark = true, ExtraValues = ExtraValues.Ignore, Delimiter = ",", Encoding = 65001]),
  PromoteHeaders = Table.PromoteHeaders(Pattern),
  TransformColumnTypes = Table.TransformColumnTypes(PromoteHeaders, columnNameAndTypePairs, [MissingField = MissingField.Ignore])
in
  TransformColumnTypes;
shared Churn_Predictions_low_TPV_DataDestination = let
  Pattern = SharePoint.Contents("https://flatpaydk.sharepoint.com/sites/Churn", [Implementation = "2.0"]),
  Navigation_1 = Pattern{[Name = "Shared Documents"]}[Content],
  FileNavigation = Navigation_1{[Name = "Churn_Predictions_low_TPV.csv"]}?[Content]?
in
  FileNavigation;
shared Churn_Predictions_low_TPV_DataDestinationTransform = (binaryStream, columnNameAndTypePairs) => let
  Pattern = Csv.Document(binaryStream, [Columns = List.Transform(columnNameAndTypePairs, each _{0}), CsvStyle = CsvStyle.QuoteAlways, IncludeByteOrderMark = true, ExtraValues = ExtraValues.Ignore, Delimiter = ",", Encoding = 65001]),
  PromoteHeaders = Table.PromoteHeaders(Pattern),
  TransformColumnTypes = Table.TransformColumnTypes(PromoteHeaders, columnNameAndTypePairs, [MissingField = MissingField.Ignore])
in
  TransformColumnTypes;
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "Churn_Predictions_medium_TPV_DataDestination", IsNewTarget = true], Settings = [Kind = "Automatic", TypeSettings = [Kind = "File", TransformationQueryName = "Churn_Predictions_medium_TPV_DataDestinationTransform"]]]}]
shared Churn_Predictions_medium_TPV = let
    // 1. Read the Delta table
    Source          = Lakehouse.Contents(null),
    #"Navigation 1" = Source{
                        [workspaceId = "e03dbe51-4f30-4e31-a84f-647f6b831f58"]
                     }[Data],
    #"Navigation 2" = #"Navigation 1"{
                        [lakehouseId = "54c8eb17-9ab5-4b14-9a01-728630a243fb"]
                     }[Data],
    #"Navigation 3" = #"Navigation 2"{
                        [Id = "Churn_Predictions_medium_TPV", ItemKind = "Table"]
                     }[Data],

    // 2. Clean up
    #"Removed columns"  = Table.RemoveColumns(
                            #"Navigation 3",
                            {"__index_level_0__"}
                          ),
    #"Changed column type" = Table.TransformColumnTypes(
                               #"Removed columns",
                               {{"phonenumber", type text}}
                             ),

   // after you’ve already cast phonenumber to text
    #"Keep plus sign" =
        Table.TransformColumns(
            #"Changed column type",
            {{"phonenumber",
              each
                  // add a zero-width space (U+200B) right after the "+"
                  if Text.StartsWith(_, "+")
                  then "+" & Character.FromNumber(8203) & Text.Range(_, 1)
                  else _,
              type text}}
        )

in
    #"Keep plus sign";
shared Churn_Predictions_medium_TPV_DataDestination = let
  Pattern = SharePoint.Contents("https://flatpaydk.sharepoint.com/sites/Churn", [Implementation = "2.0"]),
  Navigation_1 = Pattern{[Name = "Shared Documents"]}[Content],
  FileNavigation = Navigation_1{[Name = "Churn_Predictions_medium_TPV.csv"]}?[Content]?
in
  FileNavigation;
shared Churn_Predictions_medium_TPV_DataDestinationTransform = (binaryStream, columnNameAndTypePairs) => let
  Pattern = Csv.Document(binaryStream, [Columns = List.Transform(columnNameAndTypePairs, each _{0}), CsvStyle = CsvStyle.QuoteAlways, IncludeByteOrderMark = true, ExtraValues = ExtraValues.Ignore, Delimiter = ",", Encoding = 65001]),
  PromoteHeaders = Table.PromoteHeaders(Pattern),
  TransformColumnTypes = Table.TransformColumnTypes(PromoteHeaders, columnNameAndTypePairs, [MissingField = MissingField.Ignore])
in
  TransformColumnTypes;
