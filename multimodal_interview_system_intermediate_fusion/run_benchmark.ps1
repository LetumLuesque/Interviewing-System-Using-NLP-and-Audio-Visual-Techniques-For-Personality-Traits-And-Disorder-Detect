# run_benchmark.ps1 - Automated training for aggregation methods
# VERSION: 6.0 (Unified 3nd-Stage Pipeline for Mean/Weighted)

$methods = @("mean", "weighted")
$dataDir = "D:\Grad\Datasets\MDPE-Dataset"
$seumldDir = "D:\Grad\Datasets\SEUMLD\SEUMLD"
$featuresDir = "features"

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  INTERMEDIATE FUSION BENCHMARK v6.0" -ForegroundColor Cyan
Write-Host "  Mean/Weighted: 3-Stage Refine Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

foreach ($method in $methods) {
    Write-Host "`n>>> [METHOD: $method]" -ForegroundColor Green
    
    # Prepare Model Config
    $modelConfigPath = "config/benchmark/model_config_$method.yaml"
    if (!(Test-Path "config/benchmark")) { New-Item -ItemType Directory -Path "config/benchmark" | Out-Null }
    if (!(Test-Path $modelConfigPath)) {
        Copy-Item "config/model_config.yaml" $modelConfigPath
        (Get-Content $modelConfigPath) -replace 'aggregation: ".*"', "aggregation: `"$method`"" | Set-Content $modelConfigPath
    }

    # Stage 1: Personality Focus
    $s1Dir = "checkpoints/benchmark/$method/stage1"
    $s1Best = Get-ChildItem -Path "$s1Dir" -Filter "best_model_epoch_*.pt" -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (!$s1Best) {
        Write-Host "[...] Running Stage 1..." -ForegroundColor Gray
        $s1Config = "config/benchmark/training_config_${method}_stage1.yaml"
        Copy-Item "config/training_config_stage1.yaml" $s1Config
        (Get-Content $s1Config) -replace 'save_dir: "checkpoints"', "save_dir: `"$s1Dir`"" `
                                -replace 'log_dir: "logs"', "log_dir: `"logs/benchmark/$method/stage1`"" `
                                | Set-Content $s1Config
        python main_train.py --model_config $modelConfigPath --config $s1Config --data_dir "$dataDir" --seumld_data_dir "$seumldDir" --features_dir "$featuresDir" --stage 1
        $s1Best = Get-ChildItem -Path "$s1Dir" -Filter "best_model_epoch_*.pt" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    } else { Write-Host "[v] Stage 1 Completed." -ForegroundColor Green }

    # Stage 2: Deception Focus
    $s2Dir = "checkpoints/benchmark/$method/stage2"
    $s2Best = Get-ChildItem -Path "$s2Dir" -Filter "best_model_epoch_*.pt" -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (!$s2Best -and $s1Best) {
        Write-Host "[...] Running Stage 2..." -ForegroundColor Cyan
        $s2Config = "config/benchmark/training_config_${method}_stage2.yaml"
        Copy-Item "config/training_config_stage2.yaml" $s2Config
        (Get-Content $s2Config) -replace 'save_dir: "checkpoints"', "save_dir: `"$s2Dir`"" `
                                -replace 'log_dir: "logs"', "log_dir: `"logs/benchmark/$method/stage2`"" `
                                -replace 'metric: ".*"', 'metric: "val_deception_accuracy"' `
                                | Set-Content $s2Config
        python main_train.py --model_config $modelConfigPath --config $s2Config --data_dir "$dataDir" --seumld_data_dir "$seumldDir" --features_dir "$featuresDir" --use_combined --resume "$($s1Best.FullName)" --stage 2
        $s2Best = Get-ChildItem -Path "$s2Dir" -Filter "best_model_epoch_*.pt" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    } elseif ($s2Best) { Write-Host "[v] Stage 2 Completed." -ForegroundColor Green }

    # Stage 3: Joint Balance (Unified Model)
    $s3Dir = "checkpoints/benchmark/$method/stage3"
    $s3Best = Get-ChildItem -Path "$s3Dir" -Filter "best_model_epoch_*.pt" -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (!$s3Best -and $s2Best) {
        Write-Host "[...] Running Stage 3 (Joint Balance)..." -ForegroundColor Yellow
        $s3Config = "config/benchmark/training_config_${method}_stage3.yaml"
        Copy-Item "config/training_config_stage3.yaml" $s3Config
        (Get-Content $s3Config) -replace 'save_dir: "checkpoints"', "save_dir: `"$s3Dir`"" `
                                -replace 'log_dir: "logs"', "log_dir: `"logs/benchmark/$method/stage3`"" `
                                -replace 'num_epochs: \d+', "num_epochs: 120" `
                                | Set-Content $s3Config
        python main_train.py --model_config $modelConfigPath --config $s3Config --data_dir "$dataDir" --seumld_data_dir "$seumldDir" --features_dir "$featuresDir" --use_combined --resume "$($s2Best.FullName)" --stage 3
    } elseif ($s3Best) { Write-Host "[v] Stage 3 Completed." -ForegroundColor Green }
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK COMPLETED!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
