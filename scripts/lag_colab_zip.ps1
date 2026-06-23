# Lag data.zip klar for Colab (annotated + annotated_seg)
# Kjor fra prosjektmappen: .\scripts\lag_colab_zip.ps1
Set-Location $PSScriptRoot\..

$zip = "data.zip"
Write-Host "Lager $zip ..."
Compress-Archive -Path data\annotated, data\annotated_seg -DestinationPath $zip -Force

$size  = [math]::Round((Get-Item $zip).Length / 1MB, 0)
$imgs  = (Get-ChildItem data\annotated_seg\images -Recurse -File).Count
$lbls  = (Get-ChildItem data\annotated_seg\labels -Recurse -File -Filter "*.txt").Count

Write-Host ""
Write-Host "Ferdig: $zip  ($size MB)"
Write-Host "  annotated_seg: $imgs bilder, $lbls labels"
Write-Host ""
Write-Host "Last opp data.zip til roten av Google Drive."
Write-Host "Trykk Enter for aa avslutte ..."
Read-Host
