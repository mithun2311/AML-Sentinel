
$jdk17 = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

if (-not (Test-Path "$jdk17\bin\java.exe")) {
    Write-Error "Java 17 was not found at: $jdk17"
    return
}

# Change Java only in this PowerShell session and its child processes.
$env:JAVA_HOME = $jdk17
$env:PATH = "$jdk17\bin;$env:PATH"

# Tell PySpark explicitly which Python executable to use.
$pythonExe = (Get-Command python).Source
$env:PYSPARK_PYTHON = $pythonExe
$env:PYSPARK_DRIVER_PYTHON = $pythonExe

Write-Host "AML-Sentinel project environment activated." -ForegroundColor Green
Write-Host "JAVA_HOME: $env:JAVA_HOME"
Write-Host "Python: $env:PYSPARK_PYTHON"
java -version
