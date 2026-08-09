param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$KnowledgeBase = "paperops-demo",
    [int]$TimeoutSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PaperOpsUrl {
    param([string]$Path)

    $root = [System.Uri]::new($BaseUrl.TrimEnd("/") + "/")
    return [System.Uri]::new($root, $Path.TrimStart("/")).AbsoluteUri
}

function Wait-PaperOpsResult {
    param(
        [string]$StatusUrl,
        [string[]]$TerminalStatuses
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $resolvedStatusUrl = Resolve-PaperOpsUrl $StatusUrl
    do {
        $result = Invoke-RestMethod -Method Get -Uri $resolvedStatusUrl
        if (-not $result.running -and $TerminalStatuses -contains $result.status) {
            return $result
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "Timed out waiting for $resolvedStatusUrl after $TimeoutSeconds seconds."
}

function Submit-Pdf {
    param([string]$PdfPath)

    $rawResponse = & curl.exe `
        --fail-with-body `
        --silent `
        --show-error `
        -X POST `
        (Resolve-PaperOpsUrl "/jobs") `
        -F "target_knowledge_base=$KnowledgeBase" `
        -F "file=@$PdfPath;type=application/pdf"
    if ($LASTEXITCODE -ne 0) {
        throw "PDF upload failed with curl exit code $LASTEXITCODE."
    }

    $accepted = ($rawResponse -join "`n") | ConvertFrom-Json
    $completed = Wait-PaperOpsResult `
        -StatusUrl $accepted.status_url `
        -TerminalStatuses @("completed", "failed", "waiting_approval")
    if ($completed.status -ne "completed") {
        throw "PDF job ended with status '$($completed.status)': $($completed | ConvertTo-Json -Depth 10 -Compress)"
    }
    return $completed
}

$health = Invoke-RestMethod -Method Get -Uri (Resolve-PaperOpsUrl "/health")
if ($health.client_mode -ne "fake" -or $health.research_model -ne "fake-research-model") {
    throw (
        "The demo only runs against Fake mode. " +
        "Received client_mode='$($health.client_mode)', " +
        "research_model='$($health.research_model)'."
    )
}

$temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$demoDirectory = Join-Path $temporaryRoot ("paperops-demo-" + [Guid]::NewGuid().ToString("N"))
[System.IO.Directory]::CreateDirectory($demoDirectory) | Out-Null

try {
    $paperAPath = Join-Path $demoDirectory "paper-a.pdf"
    $paperBPath = Join-Path $demoDirectory "paper-b.pdf"
    [System.IO.File]::WriteAllText(
        $paperAPath,
        "%PDF-1.7`nDeterministic PaperOps Fake fixture A.",
        [System.Text.Encoding]::ASCII
    )
    [System.IO.File]::WriteAllText(
        $paperBPath,
        "%PDF-1.7`nDeterministic PaperOps Fake fixture B.",
        [System.Text.Encoding]::ASCII
    )

    $paperA = Submit-Pdf $paperAPath
    $paperB = Submit-Pdf $paperBPath

    $queryPayload = @{
        knowledge_base = $KnowledgeBase
        question = "How is the document validated, ingested, and queried?"
    } | ConvertTo-Json
    $queryAccepted = Invoke-RestMethod `
        -Method Post `
        -Uri (Resolve-PaperOpsUrl "/queries") `
        -ContentType "application/json" `
        -Body $queryPayload
    $query = Wait-PaperOpsResult `
        -StatusUrl $queryAccepted.status_url `
        -TerminalStatuses @("completed", "failed")
    if ($query.status -ne "completed") {
        throw "Research query failed: $($query | ConvertTo-Json -Depth 10 -Compress)"
    }

    $comparisonPayload = @{
        knowledge_base = $KnowledgeBase
        documents = @(
            @{
                document_id = $paperA.indexed_document_id
                label = "Paper A"
            },
            @{
                document_id = $paperB.indexed_document_id
                label = "Paper B"
            }
        )
        dimensions = @(
            @{
                dimension_id = "method"
                description = "Which method is proposed?"
            },
            @{
                dimension_id = "limitations"
                description = "Which limitations are reported?"
            }
        )
    } | ConvertTo-Json -Depth 5
    $comparisonAccepted = Invoke-RestMethod `
        -Method Post `
        -Uri (Resolve-PaperOpsUrl "/comparisons") `
        -ContentType "application/json" `
        -Body $comparisonPayload
    $comparison = Wait-PaperOpsResult `
        -StatusUrl $comparisonAccepted.status_url `
        -TerminalStatuses @("completed", "failed")
    if ($comparison.status -ne "completed") {
        throw "Comparison failed: $($comparison | ConvertTo-Json -Depth 10 -Compress)"
    }

    [ordered]@{
        mode = @{
            client = $health.client_mode
            research_model = $health.research_model
        }
        ingestion = @(
            @{
                status = $paperA.status
                document_id = $paperA.indexed_document_id
                chunks = $paperA.indexed_chunk_count
            },
            @{
                status = $paperB.status
                document_id = $paperB.indexed_document_id
                chunks = $paperB.indexed_chunk_count
            }
        )
        query = @{
            status = $query.status
            answer = $query.answer.text
            citations = @($query.answer.citation_ids)
            retrieval_calls = $query.retrieval_calls
            model_calls = $query.model_calls
        }
        comparison = @{
            status = $comparison.status
            total_cells = $comparison.total_cells
            supported_cells = $comparison.supported_cells
            missing_cells = $comparison.missing_cells
            stop_reason = $comparison.stop_reason
            retrieval_calls = $comparison.retrieval_calls
            model_calls = $comparison.model_calls
        }
    } | ConvertTo-Json -Depth 6
}
finally {
    $resolvedDemoDirectory = [System.IO.Path]::GetFullPath($demoDirectory)
    $ownedPrefix = Join-Path $temporaryRoot "paperops-demo-"
    if (
        $resolvedDemoDirectory.StartsWith(
            $ownedPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -and
        (Test-Path -LiteralPath $resolvedDemoDirectory)
    ) {
        Remove-Item -LiteralPath $resolvedDemoDirectory -Recurse -Force
    }
}
