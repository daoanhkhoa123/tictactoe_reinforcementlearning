$checkpoint = 'weights/deep_actor_critic_ep75000.pt'

if (-not (Test-Path $checkpoint)) {
    Write-Host "Checkpoint not found: $checkpoint" -ForegroundColor Yellow
    Write-Host "Current available checkpoints:" -ForegroundColor Yellow
    Get-ChildItem weights -Filter 'deep_actor_critic_ep*.pt' | Sort-Object Name | Select-Object -ExpandProperty Name
    exit 1
}

python src/rl_learning/actor_critic/play_bot.py --weights $checkpoint
