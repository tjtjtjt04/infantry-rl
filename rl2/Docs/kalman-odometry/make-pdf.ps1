param(
    [string]$OutPdf = 'D:\Robomaster\STM32Project\DM_BMI088\Docs\kalman-odometry\轮式里程计卡尔曼滤波.pdf'
)
$ErrorActionPreference = 'Stop'
$root = 'D:\Robomaster\STM32Project\DM_BMI088\Docs\kalman-odometry'
$pandoc = "$root\build\pandoc\pandoc-3.13\pandoc.exe"
$py = 'C:\Users\CMD\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$loNode = 'D:\dsh\resources\runtime\primary-runtime\dependencies\node\bin\node.exe'
$loCli = 'D:\dsh\resources\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js'

Write-Output '[1/3] pandoc: markdown -> docx (LaTeX -> OMML)'
Push-Location $root
$log = & $pandoc 'content.md' -o 'build\doc.docx' --from 'markdown+tex_math_dollars+pipe_tables' -t docx 2>&1
Pop-Location
if ($log) { $log | Select-Object -First 20 }

Write-Output '[2/3] postprocess: A4 / 封面 / 页脚 / 表格'
& $py "$root\postprocess.py" "$root\build\doc.docx" "$root\build\doc-styled.docx" 2>&1 |
    Where-Object { $_ -notmatch 'UserWarning|return d.styles' }

Write-Output '[3/3] LibreOffice Kit: docx -> pdf'
if (Test-Path $OutPdf) { Remove-Item $OutPdf -Force }
& $loNode $loCli convert --input "$root\build\doc-styled.docx" --output $OutPdf 2>&1

if (-not (Test-Path $OutPdf)) { Write-Output 'PDF_FAILED'; exit 1 }
$b = [System.IO.File]::ReadAllBytes($OutPdf)
$sig = [System.Text.Encoding]::ASCII.GetString($b, 0, 5)
$raw = [System.Text.Encoding]::Latin1.GetString($b)
$pages = [regex]::Matches($raw, '/Type\s*/Page[^s]').Count
Write-Output "PDF: $OutPdf"
Write-Output "签名: $sig ; 大小: $([math]::Round($b.Length/1KB,1)) KB ; 页数: $pages"
