# Kjor fra prosjektmappen: .\scripts\lag_colab_zip.ps1
Set-Location $PSScriptRoot\..

$zip = "data_seg.zip"
Write-Host "Lager $zip ..."
Compress-Archive -Path data\annotated_seg -DestinationPath $zip -Force

$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
$imgs = (Get-ChildItem data\annotated_seg\images -Recurse -File).Count
$lbls = (Get-ChildItem data\annotated_seg\labels -Recurse -File -Filter "*.txt").Count

Write-Host ""
Write-Host "Ferdig: $zip  ($size MB)"
Write-Host "  $imgs bilder, $lbls labels inkludert"
Write-Host ""
Write-Host "Last opp data_seg.zip til roten av Google Drive."
Write-Host "Trykk Enter for aa avslutte ..."
Read-Host
