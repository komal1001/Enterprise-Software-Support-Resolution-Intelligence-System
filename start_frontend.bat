@echo off
rem Runs Vite through node directly: npm/npx .cmd launchers break on the "&"
rem in this folder's path ("Support & Resolution").
cd /d "%~dp0frontend-react"
node node_modules\vite\bin\vite.js
pause
