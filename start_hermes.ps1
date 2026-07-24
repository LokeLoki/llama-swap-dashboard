# start_hermes.ps1
# Kill the orphans and the main agent
taskkill /F /IM "whatsapp-bridge.exe" /T 2>$null
taskkill /F /IM "python.exe" /T 2>$null
taskkill /F /IM "pythonw.exe" /T 2>$null
# 1. Gag the terminal tools so the screen doesn't strobe
$env:HERMES_DISABLE_SKILLS = "computer-use,systematic-debugging"

# 2. Start the Hermes gateway and dashboard VISIBLY
Start-Process "hermes" -ArgumentList "gateway", "run"
Start-Process "hermes" -ArgumentList "dashboard", "--host", "0.0.0.0"