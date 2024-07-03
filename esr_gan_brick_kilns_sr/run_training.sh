# Command to execute main.py with nohup
#!/bin/bash
nohup python main.py > lucknow_sarath_esrgan_fine_tuned_epochs_100.log 2>&1 &
echo "Job fired!"
