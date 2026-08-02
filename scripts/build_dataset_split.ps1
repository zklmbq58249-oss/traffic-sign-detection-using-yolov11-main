[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,
    [string]$OutputRoot = "datasets/self-driving-cars-v6",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

function Resolve-RepositoryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Get-SplitCounts {
    param([Parameter(Mandatory = $true)][int]$Total)

    $ratios = [ordered]@{
        train = 0.7
        test  = 0.2
        val   = 0.1
    }
    $counts = [ordered]@{}
    $remainders = @()
    $allocated = 0

    foreach ($split in $ratios.Keys) {
        $rawCount = $Total * $ratios[$split]
        $baseCount = [int][Math]::Floor($rawCount)
        $counts[$split] = $baseCount
        $allocated += $baseCount
        $remainders += [PSCustomObject]@{
            Split     = $split
            Remainder = $rawCount - $baseCount
        }
    }

    $remaining = $Total - $allocated
    $priority = @($remainders | Sort-Object -Property @{ Expression = "Remainder"; Descending = $true }, @{ Expression = "Split"; Descending = $false })
    for ($index = 0; $index -lt $remaining; $index++) {
        $counts[$priority[$index].Split]++
    }

    return $counts
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = Resolve-RepositoryPath -Path $SourceRoot -RepositoryRoot $repositoryRoot
$outputPath = Resolve-RepositoryPath -Path $OutputRoot -RepositoryRoot $repositoryRoot
$sourceYaml = Join-Path $sourcePath "data.yaml"

if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "Source dataset directory does not exist: $sourcePath"
}
if (-not (Test-Path -LiteralPath $sourceYaml -PathType Leaf)) {
    throw "Source data.yaml does not exist: $sourceYaml"
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output dataset directory already exists: $outputPath. Refusing to overwrite it."
}

$yamlLines = Get-Content -LiteralPath $sourceYaml
$ncLine = @($yamlLines | Where-Object { $_ -match "^\s*nc:\s*(\d+)\s*$" })[0]
$namesLine = @($yamlLines | Where-Object { $_ -match "^\s*names:\s*\[.+\]\s*$" })[0]
if (-not $ncLine -or -not $namesLine) {
    throw "Source data.yaml must define scalar 'nc' and list-form 'names' entries."
}
$classCount = [int]([regex]::Match($ncLine, "^\s*nc:\s*(\d+)\s*$").Groups[1].Value)

$supportedExtensions = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($extension in ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp") {
    [void]$supportedExtensions.Add($extension)
}

$items = [System.Collections.Generic.List[object]]::new()
$seenNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($sourceSplit in "train", "valid", "test") {
    $imagesPath = Join-Path $sourcePath "$sourceSplit\images"
    $labelsPath = Join-Path $sourcePath "$sourceSplit\labels"
    if (-not (Test-Path -LiteralPath $imagesPath -PathType Container)) {
        throw "Source image directory does not exist: $imagesPath"
    }
    if (-not (Test-Path -LiteralPath $labelsPath -PathType Container)) {
        throw "Source label directory does not exist: $labelsPath"
    }

    $images = @(Get-ChildItem -LiteralPath $imagesPath -File | Where-Object { $supportedExtensions.Contains($_.Extension) } | Sort-Object Name)
    foreach ($image in $images) {
        $label = Join-Path $labelsPath ("{0}.txt" -f $image.BaseName)
        if (-not (Test-Path -LiteralPath $label -PathType Leaf)) {
            throw "Missing label for source image: $($image.FullName)"
        }
        if (-not $seenNames.Add($image.Name)) {
            throw "Duplicate image filename across source splits: $($image.Name)"
        }
        $items.Add([PSCustomObject]@{
            Image       = $image
            Label       = Get-Item -LiteralPath $label
            SourceSplit = $sourceSplit
        })
    }
}

if ($items.Count -eq 0) {
    throw "No supported images were found in the source dataset."
}

$random = [System.Random]::new($Seed)
for ($index = $items.Count - 1; $index -gt 0; $index--) {
    $swapIndex = $random.Next($index + 1)
    $temporary = $items[$index]
    $items[$index] = $items[$swapIndex]
    $items[$swapIndex] = $temporary
}

$splitCounts = Get-SplitCounts -Total $items.Count
New-Item -ItemType Directory -Path $outputPath | Out-Null
foreach ($split in "train", "val", "test") {
    New-Item -ItemType Directory -Path (Join-Path $outputPath "$split\images") | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $outputPath "$split\labels") | Out-Null
}

$manifest = [System.Collections.Generic.List[object]]::new()
$offset = 0
foreach ($split in "train", "test", "val") {
    $count = [int]$splitCounts[$split]
    for ($index = $offset; $index -lt ($offset + $count); $index++) {
        $item = $items[$index]
        Copy-Item -LiteralPath $item.Image.FullName -Destination (Join-Path $outputPath "$split\images\$($item.Image.Name)")
        Copy-Item -LiteralPath $item.Label.FullName -Destination (Join-Path $outputPath "$split\labels\$($item.Label.Name)")
        $manifest.Add([PSCustomObject]@{
            split        = $split
            image        = $item.Image.Name
            label        = $item.Label.Name
            source_split = $item.SourceSplit
        })
    }
    $offset += $count
}

$dataYaml = @(
    "# Rebuilt from the Roboflow v1 export with a deterministic 7:2:1 train:test:val split.",
    "path: .",
    "train: train/images",
    "val: val/images",
    "test: test/images",
    "",
    $ncLine.Trim(),
    $namesLine.Trim()
)
$dataYaml | Set-Content -LiteralPath (Join-Path $outputPath "data.yaml") -Encoding utf8
$manifest | Export-Csv -LiteralPath (Join-Path $outputPath "split_manifest.csv") -NoTypeInformation -Encoding utf8

$report = [ordered]@{
    source_root  = $sourcePath
    output_root  = $outputPath
    seed         = $Seed
    total_images = $items.Count
    split_order  = "train:test:val"
    split_counts = [ordered]@{
        train = $splitCounts.train
        test  = $splitCounts.test
        val   = $splitCounts.val
    }
    class_count  = $classCount
}
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $outputPath "split_report.json") -Encoding utf8

Write-Host "Built $($items.Count) image/label pairs at $outputPath"
Write-Host "train=$($splitCounts.train), test=$($splitCounts.test), val=$($splitCounts.val), seed=$Seed"
