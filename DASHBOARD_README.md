# OSU AI Training Dashboard

A real-time web dashboard for monitoring the training progress of the OSU AI Replay Maker.

## Features

- **Real-time Training Monitoring**: View current epoch, batch progress, and training status
- **Loss Visualization**: Interactive graphs showing training loss over time
- **Training Metrics**: Current learning rate, best loss, estimated time remaining
- **Modern UI**: Clean, responsive design using Bootstrap
- **Auto-refresh**: Updates every 2 seconds automatically

## Quick Start

### Option 1: Run Dashboard Separately (Recommended)

1. **Start the dashboard** (in one terminal):
   ```bash
   python run_dashboard.py
   ```
   
2. **Start training** (in another terminal):
   ```bash
   python train.py
   ```

3. **Open your browser** and go to: http://localhost:5000

### Option 2: Manual Setup

1. **Install Flask** (if not already installed):
   ```bash
   pip install flask
   ```

2. **Run the dashboard**:
   ```bash
   python dashboard.py
   ```

3. **Run training** (in another terminal):
   ```bash
   python train.py
   ```

## Dashboard Components

### Status Cards
- **Current Epoch**: Shows the current training epoch
- **Best Loss**: Displays the lowest loss achieved so far
- **Learning Rate**: Current learning rate value
- **Status**: Training status (Not Started, Training, etc.)

### Progress Bars
- **Epoch Progress**: Overall training progress across all epochs
- **Batch Progress**: Progress within the current epoch

### Loss Chart
- Real-time line chart showing training loss over time
- Automatically updates as new data comes in
- Shows last 500 data points for performance

### Training Information
- **Last Update**: When the dashboard last received new data
- **Estimated Time Remaining**: Calculated based on current progress
- **Current Learning Rate**: Real-time learning rate value

### System Status
- **Dashboard Status**: Shows if the dashboard is online
- **Auto Refresh**: Indicates refresh interval
- **Data Points**: Number of loss data points collected

## How It Works

1. **Log Monitoring**: The dashboard monitors the `training.log` file for new entries
2. **Log Parsing**: It parses training logs to extract epoch, batch, loss, and learning rate information
3. **Real-time Updates**: The web interface polls the backend every 2 seconds for updates
4. **Data Visualization**: Uses Chart.js for interactive loss graphs

## Log Format

The dashboard expects log entries in this format:
```
2024-01-15 10:30:45 - Epoch 1/100 - Train Loss: 0.123456 - Val Loss: 0.234567 - LR: 1.00e-04
2024-01-15 10:30:50 - Batch 10/500 - Loss: 0.123456 - LR: 1.00e-04
```

## Troubleshooting

### Dashboard shows "Not Started"
- Make sure training is running and generating logs
- Check that `training.log` exists in the same directory
- Verify the log format matches what the dashboard expects

### No loss data in chart
- Ensure training has started and is logging loss values
- Check the browser console for any JavaScript errors
- Verify the `/api/loss_data` endpoint is returning data

### Dashboard not updating
- Check that the training script is actively writing to `training.log`
- Refresh the browser page
- Check the dashboard console output for errors

### Port 5000 already in use
- Change the port in `dashboard.py`:
  ```python
  app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
  ```

## Customization

### Changing Update Interval
Modify the refresh interval in `templates/dashboard.html`:
```javascript
setInterval(() => {
    updateDashboard();
    updateChart();
}, 5000); // Update every 5 seconds instead of 2
```

### Adding Custom Metrics
1. Modify the log parsing in `dashboard.py`
2. Add new API endpoints for your metrics
3. Update the HTML template to display them

### Styling
The dashboard uses Bootstrap 5 and custom CSS. Modify the `<style>` section in `templates/dashboard.html` to customize the appearance.

## Files

- `dashboard.py`: Main Flask application
- `templates/dashboard.html`: Web interface template
- `run_dashboard.py`: Helper script to start the dashboard
- `training.log`: Log file monitored by the dashboard (created by training)

## Requirements

- Python 3.7+
- Flask
- Modern web browser
- Training script that outputs compatible log format

## Tips

- Keep the dashboard running in a separate terminal while training
- The dashboard works best with consistent log output from training
- Use a second monitor to keep the dashboard visible while working
- The loss chart automatically scales and shows the most recent data
- All data is stored in memory, so restarting the dashboard will reset the history